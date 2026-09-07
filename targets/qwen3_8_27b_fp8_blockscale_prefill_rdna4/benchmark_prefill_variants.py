#!/usr/bin/env python3
"""Directly benchmark prefill kernel variants over a generalization grid.

The candidate router is intentionally bypassed: baseline, BM32, BM64, and BM80
are launched directly on identical tensors.  The output is a raw receipt for
``select_prefill_routes.py``; this script never decides a route itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch
import triton

HERE = Path(__file__).resolve().parent
GEAK_ROOT = HERE.parents[1]
DEFAULT_CANDIDATE = HERE / "variants/round4_kernel.py"
DEFAULT_CASES = HERE / "baseline/cases.json"
DEFAULT_OUTPUT = HERE / "capture/prefill_variant_bench.json"
DEFAULT_M_VALUES = (
    32,
    33,
    63,
    64,
    72,
    80,
    96,
    128,
    129,
    138,
    139,
    192,
    224,
    225,
    240,
    249,
    256,
    257,
    277,
    384,
    512,
    513,
    523,
    640,
    641,
    768,
    783,
    784,
    785,
    800,
    801,
    832,
    1024,
)
VARIANT_SPECS = {
    "bm32": ("_prefill_bm32_bn128", 32),
    "bm64": ("_prefill_bm64_bn128_fused_scale", 64),
    "bm80": ("_prefill_bm80_bn128_shared_b", 80),
}


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def _parse_variants(value: str) -> list[str]:
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(VARIANT_SPECS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}")
    return list(dict.fromkeys(result))


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        n_text, k_text = value.lower().split("x", 1)
        result = int(n_text), int(k_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be NxK") from exc
    if min(result) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return result


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "geak_prefill_selection_candidate", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, (symbol, _) in VARIANT_SPECS.items():
        if not hasattr(module, symbol):
            raise RuntimeError(f"candidate lacks {name} symbol {symbol}: {path}")
    return module


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride = 1
    result = []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def _make_tensor(
    shape: list[int],
    stride: list[int],
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        value = (torch.randn(shape, device="cuda", generator=generator) * 0.25).to(
            dtype
        )
    elif dtype == torch.float32:
        value = torch.rand(shape, device="cuda", generator=generator) * 0.05 + 0.001
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    if stride == _contiguous_stride(shape):
        return value
    result = torch.empty_strided(shape, stride, device="cuda", dtype=dtype)
    result.copy_(value)
    return result


def _templates(
    cases_path: Path,
) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int, int], float]]:
    cases = json.loads(cases_path.read_text())["cases"]
    templates: dict[tuple[int, int], dict] = {}
    production_weights: dict[tuple[int, int, int], float] = {}
    for case in cases:
        n, k, m = int(case["N"]), int(case["K"]), int(case["M"])
        if case.get("regime") == "prefill":
            templates.setdefault((n, k), case)
        if case.get("coverage_role") == "scored_captured_prefill":
            production_weights[(m, n, k)] = float(case["weight"])
    return templates, production_weights


def _make_inputs(template: dict, m: int, seed: int) -> tuple[torch.Tensor, ...]:
    k = int(template["K"])
    k_blocks = k // 128
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        _make_tensor([m, k], [k, 1], torch.float8_e4m3fn, generator),
        _make_tensor(
            template["b_shape"], template["b_stride"], torch.float8_e4m3fn, generator
        ),
        _make_tensor([m, k_blocks], [k_blocks, 1], torch.float32, generator),
        _make_tensor(
            template["bs_shape"], template["bs_stride"], torch.float32, generator
        ),
    )


def _baseline(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    activation, weight, activation_scale, weight_scale = inputs
    return w8a8_triton_block_scaled_mm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        [128, 128],
        torch.bfloat16,
    )


def _variant_call(
    candidate: ModuleType,
    variant: str,
    inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    symbol, block_m = VARIANT_SPECS[variant]
    kernel = getattr(candidate, symbol)
    activation, weight, activation_scale, weight_scale = inputs
    m, k = activation.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, 128),)
    kernel[grid](
        activation,
        weight,
        output,
        activation_scale,
        weight_scale,
        m,
        n,
        k,
        activation.stride(0),
        weight.stride(0),
        activation_scale.stride(0),
        activation_scale.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        GROUP_M=32,
        num_warps=4,
        num_stages=2,
    )
    return output


def _correctness(
    candidate: ModuleType,
    variants: list[str],
    template: dict,
    m: int,
    seed: int,
    draws: int,
    rtol: float,
    atol: float,
) -> dict[str, dict]:
    result = {
        name: {
            "correct": True,
            "independent_output": True,
            "max_abs_error": 0.0,
            "error": "",
        }
        for name in variants
    }
    for draw in range(draws):
        inputs = _make_inputs(template, m, seed + draw * 1_000_003)
        reference = _baseline(inputs).detach().clone()
        for name in variants:
            if not result[name]["correct"]:
                continue
            try:
                actual = _variant_call(candidate, name, inputs)
                torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
                error = float((actual.float() - reference.float()).abs().max().item())
                result[name]["max_abs_error"] = max(
                    result[name]["max_abs_error"], error
                )
            except Exception as exc:
                result[name]["correct"] = False
                result[name]["error"] = f"{type(exc).__name__}: {exc}"
        del reference, inputs

    timing_inputs = _make_inputs(template, m, seed + 9_000_019)
    for name in variants:
        if not result[name]["correct"]:
            continue
        try:
            first = _variant_call(candidate, name, timing_inputs)
            second = _variant_call(candidate, name, timing_inputs)
            torch.cuda.synchronize()
            result[name]["independent_output"] = first.data_ptr() != second.data_ptr()
            if not result[name]["independent_output"]:
                result[name]["correct"] = False
                result[name]["error"] = "reused output storage"
        except Exception as exc:
            result[name]["correct"] = False
            result[name]["error"] = f"independence check: {type(exc).__name__}: {exc}"
    del timing_inputs
    return result


def _timings(
    candidate: ModuleType,
    variants: list[str],
    inputs: tuple[torch.Tensor, ...],
    case_index: int,
    passes: int,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, list[dict]], list[list[str]]]:
    scripts = GEAK_ROOT / "e2e_workflow/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from harness_lib import time_op

    calls: dict[str, Callable[[], torch.Tensor]] = {
        "baseline": lambda: _baseline(inputs),
        **{
            name: (lambda variant=name: _variant_call(candidate, variant, inputs))
            for name in variants
        },
    }
    names = list(calls)
    results = {name: [] for name in names}
    orders = []
    for pass_index in range(passes):
        shift = (case_index + pass_index) % len(names)
        order = names[shift:] + names[:shift]
        if pass_index % 2:
            order = list(reversed(order))
        orders.append(order)
        for name in order:
            detail = time_op(
                calls[name],
                warmup=warmup,
                repeats=repeats,
                inner=1,
                graph=False,
                flush_cache=True,
                detail=True,
            )
            if detail is None:
                raise RuntimeError(f"timing failed: {name}")
            results[name].append(detail)
    return results, orders


def _summarize_timing(
    timings: dict[str, list[dict]], variants: list[str]
) -> dict[str, dict]:
    baseline_ms = [float(item["ms"]) for item in timings["baseline"]]
    result = {
        "baseline": {
            "passes": timings["baseline"],
            "median_ms": statistics.median(baseline_ms),
            "spread": max(baseline_ms) / min(baseline_ms) - 1.0,
        }
    }
    for name in variants:
        current_ms = [float(item["ms"]) for item in timings[name]]
        speedups = [base / current for base, current in zip(baseline_ms, current_ms)]
        result[name] = {
            "passes": timings[name],
            "median_ms": statistics.median(current_ms),
            "spread": max(current_ms) / min(current_ms) - 1.0,
            "speedup": statistics.median(baseline_ms) / statistics.median(current_ms),
            "pass_speedups": speedups,
            "conservative_speedup": min(speedups),
        }
    return result


def _checkpoint(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m-values", type=_parse_ints, default=list(DEFAULT_M_VALUES))
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--variants", type=_parse_variants, default=list(VARIANT_SPECS))
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--correctness-draws", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--cache-flush-mb", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.passes, args.warmup, args.repeats, args.correctness_draws) <= 0:
        raise SystemExit(
            "passes, warmup, repeats, and correctness draws must be positive"
        )
    os.environ["HARNESS_CACHE_FLUSH_MB"] = str(args.cache_flush_mb)
    candidate_path = args.candidate.resolve()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    cases_path = args.cases.resolve()
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    candidate = _load_module(candidate_path)
    templates, production_weights = _templates(cases_path)
    if args.shape:
        wanted = set(args.shape)
        templates = {key: value for key, value in templates.items() if key in wanted}
        if set(templates) != wanted:
            raise SystemExit(f"unknown N/K families: {sorted(wanted - set(templates))}")

    config = {
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "cases": str(cases_path),
        "cases_sha256": cases_sha,
        "m_values": args.m_values,
        "families": [list(key) for key in sorted(templates)],
        "variants": args.variants,
        "passes": args.passes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness_draws": args.correctness_draws,
        "rtol": args.rtol,
        "atol": args.atol,
        "cache_flush_mb": args.cache_flush_mb,
        "paired_pass_index": True,
        "rotated_reversed_order": True,
    }
    report = {
        "schema": "geak.prefill_variant_bench.v1",
        "status": "running",
        "device": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "software": {
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
        },
        "config": config,
        "rows": [],
    }
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text())
        if existing.get("config") != config:
            raise SystemExit("cannot resume: benchmark configuration changed")
        report = existing
        report["status"] = "running"
    completed = {row["sig"] for row in report["rows"]}

    case_index = 0
    for (n, k), template in sorted(templates.items()):
        for m in args.m_values:
            sig = f"prefill_select_m{m}_n{n}_k{k}"
            if sig in completed:
                case_index += 1
                continue
            seed = 1_700_000 + m * 101 + n + k
            print(f"\n{sig}", flush=True)
            correctness = _correctness(
                candidate,
                args.variants,
                template,
                m,
                seed,
                args.correctness_draws,
                args.rtol,
                args.atol,
            )
            valid_variants = [
                name for name in args.variants if correctness[name]["correct"]
            ]
            timing_inputs = _make_inputs(template, m, seed + 7_000_033)
            timings, orders = _timings(
                candidate,
                valid_variants,
                timing_inputs,
                case_index,
                args.passes,
                args.warmup,
                args.repeats,
            )
            summary = _summarize_timing(timings, valid_variants)
            for name in args.variants:
                if name in summary:
                    summary[name].update(correctness[name])
                else:
                    summary[name] = correctness[name]
            row = {
                "sig": sig,
                "M": m,
                "N": n,
                "K": k,
                "production_weight": production_weights.get((m, n, k), 0.0),
                "order_by_pass": orders,
                "results": summary,
            }
            report["rows"].append(row)
            _checkpoint(args.output, report)
            best = max(
                ((summary[name].get("speedup", 0.0), name) for name in valid_variants),
                default=(0.0, "none"),
            )
            print(
                f"  baseline={summary['baseline']['median_ms']:.6f} ms "
                f"best={best[1]} {best[0]:.4f}x",
                flush=True,
            )
            del timing_inputs
            torch.cuda.empty_cache()
            case_index += 1

    report["status"] = "complete"
    report["num_cases"] = len(report["rows"])
    report["all_correct"] = all(
        row["results"][name].get("correct", False)
        for row in report["rows"]
        for name in args.variants
    )
    _checkpoint(args.output, report)
    print(
        f"\nWrote {args.output}: {report['num_cases']} cases, all_correct={report['all_correct']}"
    )


if __name__ == "__main__":
    main()
