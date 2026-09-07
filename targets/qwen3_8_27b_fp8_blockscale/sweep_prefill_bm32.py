#!/usr/bin/env python3
"""Compare the incumbent FP8 GEMM with the winning BM32 prefill kernel.

This driver intentionally launches ``_prefill_bm32_bn128`` directly.  Going
through the candidate's ``run`` function would silently benchmark the incumbent
for every shape outside the existing seven-signature routing rule.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import torch
import triton

HERE = Path(__file__).resolve().parent
GEAK_ROOT = HERE.parents[1]
DEFAULT_CANDIDATE = (
    GEAK_ROOT
    / "exp/bakeoff_baseline_20260822_102440/bakeoff/triton"
    / "team_task_20260822_103835_290494_26777/task/round_4/engineer_1"
    / "workspace/kernel_src/kernel.py"
)

# Powers of two establish the broad trend.  Captured M values and both sides of
# the proposed exclusive 32 < M < 784 interval catch boundary reversals.
DEFAULT_M_VALUES = (
    16,
    31,
    32,
    33,
    64,
    72,
    128,
    138,
    139,
    249,
    256,
    277,
    512,
    523,
    768,
    783,
    784,
    785,
    1024,
)


def _load_candidate(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("geak_rdna4_winner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "_prefill_bm32_bn128"):
        raise RuntimeError(f"candidate has no _prefill_bm32_bn128: {path}")
    return module


def _parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("M values must be positive integers")
    return result


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        n_text, k_text = value.lower().split("x", 1)
        shape = (int(n_text), int(k_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be NxK, e.g. 5120x3072") from exc
    if min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


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
        value = (
            torch.rand(shape, device="cuda", generator=generator, dtype=dtype) * 0.05
            + 0.001
        )
    else:
        raise ValueError(f"unsupported sweep dtype: {dtype}")
    if stride == _contiguous_stride(shape):
        return value
    result = torch.empty_strided(shape, stride, device="cuda", dtype=dtype)
    result.copy_(value)
    return result


def _weight_templates(cases_path: Path) -> list[dict]:
    cases = json.loads(cases_path.read_text())["cases"]
    templates: dict[tuple[int, int], dict] = {}
    for case in cases:
        key = (int(case["N"]), int(case["K"]))
        templates.setdefault(key, case)
    return [templates[key] for key in sorted(templates)]


def _make_weights(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(int(case["seed"]))
    weight = _make_tensor(
        case["b_shape"],
        case["b_stride"],
        torch.float8_e4m3fn,
        generator,
    )
    weight_scale = _make_tensor(
        case["bs_shape"],
        case["bs_stride"],
        torch.float32,
        generator,
    )
    return weight, weight_scale


def _make_activation(m: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed + m * 1009)
    activation = _make_tensor([m, k], [k, 1], torch.float8_e4m3fn, generator)
    activation_scale = _make_tensor(
        [m, triton.cdiv(k, 128)],
        [triton.cdiv(k, 128), 1],
        torch.float32,
        generator,
    )
    return activation, activation_scale


def _run_bm32(
    candidate: ModuleType,
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = activation.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 32) * triton.cdiv(n, 128),)
    candidate._prefill_bm32_bn128[grid](
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


def _run_baseline(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        [128, 128],
        torch.bfloat16,
    )


def _time(
    call,
    *,
    warmup: int,
    repeats: int,
    flush_cache: bool,
) -> dict:
    scripts = GEAK_ROOT / "e2e_workflow/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from harness_lib import time_op

    result = time_op(
        call,
        warmup=warmup,
        repeats=repeats,
        inner=1,
        graph=False,
        flush_cache=flush_cache,
        detail=True,
    )
    if result is None:
        raise RuntimeError("timed kernel launch failed")
    return result


def _sweep_one(
    candidate: ModuleType,
    case: dict,
    m: int,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    n, k = int(case["N"]), int(case["K"])
    activation, activation_scale = _make_activation(m, k, int(case["seed"]))
    baseline = lambda: _run_baseline(activation, weight, activation_scale, weight_scale)
    bm32 = lambda: _run_bm32(
        candidate, activation, weight, activation_scale, weight_scale
    )

    reference = baseline()
    actual = bm32()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual,
        reference,
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=False,
    )
    max_abs_error = float((actual.float() - reference.float()).abs().max().item())

    # Repeat the pair in reverse order to expose first/second-run bias.  The
    # reported latency is the median of the two independently timed medians.
    baseline_first = _time(
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
        flush_cache=not args.no_cache_flush,
    )
    bm32_second = _time(
        bm32,
        warmup=args.warmup,
        repeats=args.repeats,
        flush_cache=not args.no_cache_flush,
    )
    bm32_first = _time(
        bm32,
        warmup=args.warmup,
        repeats=args.repeats,
        flush_cache=not args.no_cache_flush,
    )
    baseline_second = _time(
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
        flush_cache=not args.no_cache_flush,
    )
    baseline_ms = statistics.median([baseline_first["ms"], baseline_second["ms"]])
    bm32_ms = statistics.median([bm32_first["ms"], bm32_second["ms"]])
    return {
        "M": m,
        "N": n,
        "K": k,
        "baseline_ms": baseline_ms,
        "bm32_ms": bm32_ms,
        "speedup": baseline_ms / bm32_ms,
        "max_abs_error": max_abs_error,
        "timing_order": {
            "baseline_first": baseline_first,
            "bm32_second": bm32_second,
            "bm32_first": bm32_first,
            "baseline_second": baseline_second,
        },
    }


def _summarize(rows: list[dict], min_speedup: float) -> list[dict]:
    by_m: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_m[row["M"]].append(row)
    summary = []
    for m, group in sorted(by_m.items()):
        speedups = [row["speedup"] for row in group]
        summary.append(
            {
                "M": m,
                "min_speedup": min(speedups),
                "geomean_speedup": math.prod(speedups) ** (1.0 / len(speedups)),
                "max_speedup": max(speedups),
                "passing_shapes": sum(value >= min_speedup for value in speedups),
                "total_shapes": len(speedups),
                "all_shapes_pass": all(value >= min_speedup for value in speedups),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "baseline/cases.json")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--m-values",
        type=_parse_ints,
        default=list(DEFAULT_M_VALUES),
        help="comma-separated M values",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="limit to a captured NxK weight shape; repeat as needed",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--no-cache-flush", action="store_true")
    parser.add_argument("--cache-flush-mb", type=int, default=512)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.01,
        help="minimum per-shape speedup for a sampled M to be route-safe",
    )
    parser.add_argument("--range-lower", type=int, default=32)
    parser.add_argument("--range-upper", type=int, default=784)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["HARNESS_CACHE_FLUSH_MB"] = str(args.cache_flush_mb)
    templates = _weight_templates(args.cases)
    if args.shape:
        wanted = set(args.shape)
        templates = [
            case for case in templates if (int(case["N"]), int(case["K"])) in wanted
        ]
        found = {(int(case["N"]), int(case["K"])) for case in templates}
        if found != wanted:
            raise SystemExit(f"unknown captured shapes: {sorted(wanted - found)}")
    candidate = _load_candidate(args.candidate.resolve())

    rows = []
    for case in templates:
        n, k = int(case["N"]), int(case["K"])
        print(f"\nN={n} K={k}", flush=True)
        weight, weight_scale = _make_weights(case)
        for m in args.m_values:
            row = _sweep_one(candidate, case, m, weight, weight_scale, args)
            rows.append(row)
            print(
                f"  M={m:4d}  baseline={row['baseline_ms']:.5f} ms"
                f"  bm32={row['bm32_ms']:.5f} ms"
                f"  speedup={row['speedup']:.4f}x",
                flush=True,
            )

    summary = _summarize(rows, args.min_speedup)
    print("\nCross-shape summary", flush=True)
    for item in summary:
        print(
            f"  M={item['M']:4d}  min={item['min_speedup']:.4f}x"
            f"  geo={item['geomean_speedup']:.4f}x"
            f"  max={item['max_speedup']:.4f}x"
            f"  pass={item['passing_shapes']}/{item['total_shapes']}",
            flush=True,
        )
    sampled_range = [
        item for item in summary if args.range_lower < item["M"] < args.range_upper
    ]
    proposed_range_safe = bool(sampled_range) and all(
        item["all_shapes_pass"] for item in sampled_range
    )
    print(
        f"\nSampled {args.range_lower} < M < {args.range_upper} route safe at"
        f" {args.min_speedup:.3f}x: {proposed_range_safe}",
        flush=True,
    )

    report = {
        "schema": "geak.rdna4_bm32_prefill_sweep.v1",
        "candidate": str(args.candidate.resolve()),
        "cases": str(args.cases.resolve()),
        "device": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "paired_reverse_order": True,
            "cache_flush": not args.no_cache_flush,
            "cache_flush_mb": args.cache_flush_mb,
            "min_speedup": args.min_speedup,
        },
        "proposed_exclusive_range": [args.range_lower, args.range_upper],
        "sampled_proposed_range_safe": proposed_range_safe,
        "rows": rows,
        "summary_by_m": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
