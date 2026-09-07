#!/usr/bin/env python3
"""Compare the RDNA4 hybrid GEMM with the colleague R9700 Triton branch."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

HERE = Path(__file__).resolve().parent
OUR_RDNA4 = Path("/app/vllm/vllm/model_executor/kernels/linear/scaled_mm/rdna4.py")
HARNESS = Path(
    "/app/GEAK/exp/bakeoff_baseline_20260822_102440/bakeoff/triton/"
    "team_task_20260822_103835_290494_26777/task/round_4/engineer_1/workspace"
)
TUNED_QUANTIZED_SHAPES = (
    (2048, 5120),
    (3584, 5120),
    (4096, 5120),
    (4352, 5120),
    (5120, 4352),
    (8704, 5120),
)
TUNED_FUSED_SHAPES = (
    (2048, 5120),
    (4352, 5120),
    (5120, 2176),
    (5120, 4352),
    (5120, 768),
)
TIMING_GRAPH = False


def _load_implementations():
    import vllm._rocm_C  # noqa: F401
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_w8a8_block_fp8_configs,
        get_w8a8_block_fp8_unquantized_configs,
        per_token_group_quant_fp8,
        w8a8_triton_block_scaled_mm,
        w8a8_triton_block_scaled_mm_unquantized,
    )

    ops.rdna4_fp8_block_scaled_mm_decode = (
        torch.ops._rocm_C.rdna4_fp8_block_scaled_mm_decode
    )
    module_name = "vllm.model_executor.kernels.linear.scaled_mm.rdna4_ours"
    spec = importlib.util.spec_from_file_location(module_name, OUR_RDNA4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OUR_RDNA4}")
    ours = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ours)
    return {
        "ours": ours,
        "colleague_quantized": w8a8_triton_block_scaled_mm,
        "colleague_unquantized": w8a8_triton_block_scaled_mm_unquantized,
        "quantize": per_token_group_quant_fp8,
        "quantized_configs": get_w8a8_block_fp8_configs,
        "fused_configs": get_w8a8_block_fp8_unquantized_configs,
    }


def _dtype(name: str) -> torch.dtype:
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _make_tensor(case: dict, stem: str, gen: torch.Generator) -> torch.Tensor:
    shape = case[f"{stem}_shape"]
    stride = case[f"{stem}_stride"]
    dtype = _dtype(case[f"{stem}_dtype"])
    if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        value = (torch.randn(shape, device="cuda", generator=gen) * 0.25).to(dtype)
    elif dtype == torch.float32:
        value = (
            torch.rand(shape, device="cuda", generator=gen, dtype=dtype) * 0.05 + 0.001
        )
    else:
        value = torch.randn(shape, device="cuda", generator=gen, dtype=dtype) * 0.25
    if tuple(value.stride()) == tuple(stride):
        return value
    result = torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
    result.copy_(value)
    return result


def _make_case_inputs(case: dict) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(int(case["seed"]))
    return tuple(_make_tensor(case, stem, gen) for stem in ("a", "b", "as", "bs"))


def _make_synthetic_inputs(m: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn((m, k), device="cuda", generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    a_scale = (
        torch.rand((m, math.ceil(k / 128)), device="cuda", generator=gen) * 0.05 + 0.001
    )
    b_scale = (
        torch.rand(
            (math.ceil(n / 128), math.ceil(k / 128)), device="cuda", generator=gen
        )
        * 0.05
        + 0.001
    )
    return a, b, a_scale, b_scale


def _timed(call: Callable[[], torch.Tensor], warmup: int, repeats: int) -> dict:
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from harness_lib import time_op

    result = time_op(
        call,
        warmup=warmup,
        repeats=repeats,
        inner=1,
        graph=TIMING_GRAPH,
        flush_cache=True,
        detail=True,
    )
    if result is None:
        raise RuntimeError("timed kernel failed")
    return result


def _compare_pair(
    ours_call: Callable[[], torch.Tensor],
    colleague_call: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    rtol: float,
    atol: float,
) -> dict:
    ours_out = ours_call()
    colleague_out = colleague_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(ours_out, colleague_out, rtol=rtol, atol=atol)
    max_abs = float((ours_out.float() - colleague_out.float()).abs().max().item())

    ours_first = _timed(ours_call, warmup, repeats)
    colleague_second = _timed(colleague_call, warmup, repeats)
    colleague_first = _timed(colleague_call, warmup, repeats)
    ours_second = _timed(ours_call, warmup, repeats)
    ours_ms = statistics.median([ours_first["ms"], ours_second["ms"]])
    colleague_ms = statistics.median([colleague_first["ms"], colleague_second["ms"]])
    return {
        "ours_ms": ours_ms,
        "colleague_ms": colleague_ms,
        "ours_speedup": colleague_ms / ours_ms,
        "winner": "ours" if ours_ms < colleague_ms else "colleague",
        "max_abs_error": max_abs,
        "timing_order": {
            "ours_first": ours_first,
            "colleague_second": colleague_second,
            "colleague_first": colleague_first,
            "ours_second": ours_second,
        },
    }


def _our_route(ours, m: int, n: int, k: int) -> str:
    if m in (1, 2) and k % 256 == 0:
        return "native_hip_decode"
    if ours.should_use_rdna4_bm32(m, n, k):
        return "triton_bm32_prefill"
    return "baseline_fallback"


def _summarize(rows: list[dict], weight_key: str = "call_count") -> dict:
    result = {}
    regimes = sorted({row["regime"] for row in rows}) + ["all"]
    for regime in regimes:
        selected = (
            rows if regime == "all" else [r for r in rows if r["regime"] == regime]
        )
        ours_total = sum(r["ours_ms"] * r.get(weight_key, 1) for r in selected)
        colleague_total = sum(
            r["colleague_ms"] * r.get(weight_key, 1) for r in selected
        )
        result[regime] = {
            "cases": len(selected),
            "ours_wins": sum(r["winner"] == "ours" for r in selected),
            "colleague_wins": sum(r["winner"] == "colleague" for r in selected),
            "ours_weighted_ms": ours_total,
            "colleague_weighted_ms": colleague_total,
            "ours_weighted_speedup": colleague_total / ours_total,
        }
    return result


def _run_exact_quantized(impls, cases: list[dict], args) -> list[dict]:
    rows = []
    for index, case in enumerate(cases, 1):
        a, b, a_scale, b_scale = _make_case_inputs(case)

        def ours_call(a=a, b=b, a_scale=a_scale, b_scale=b_scale):
            return impls["ours"]._rdna4_fp8_block_scaled_mm_impl(a, b, a_scale, b_scale)

        def colleague_call(a=a, b=b, a_scale=a_scale, b_scale=b_scale):
            return impls["colleague_quantized"](
                a, b, a_scale, b_scale, [128, 128], torch.bfloat16
            )

        row = _compare_pair(
            ours_call,
            colleague_call,
            args.warmup,
            args.repeats,
            args.rtol,
            args.atol,
        )
        row.update(
            sig=case["sig"],
            M=case["M"],
            N=case["N"],
            K=case["K"],
            regime=case["regime"],
            call_count=case["call_count"],
            our_route=_our_route(impls["ours"], case["M"], case["N"], case["K"]),
            colleague_route=(
                "tuned_config"
                if impls["quantized_configs"](case["N"], case["K"], 128, 128)
                else "baseline_fallback"
            ),
        )
        rows.append(row)
        print(
            f"exact {index:02d}/{len(cases)} {case['M']}x{case['N']}x{case['K']} "
            f"ours={row['ours_ms']:.5f}ms colleague={row['colleague_ms']:.5f}ms "
            f"speedup={row['ours_speedup']:.3f}x",
            flush=True,
        )
        del a, b, a_scale, b_scale
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _run_exact_decode_e2e(impls, cases: list[dict], args) -> list[dict]:
    rows = []
    decode_cases = [case for case in cases if case["M"] in (1, 2)]
    for index, case in enumerate(decode_cases, 1):
        _, b, _, b_scale = _make_case_inputs(case)
        gen = torch.Generator(device="cuda").manual_seed(int(case["seed"]) + 999)
        a_bf16 = torch.randn(
            (case["M"], case["K"]),
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )

        def ours_call(a_bf16=a_bf16, b=b, b_scale=b_scale):
            a, a_scale = impls["quantize"](
                a_bf16, 128, dtype=torch.float8_e4m3fn, use_ue8m0=False
            )
            return impls["ours"]._rdna4_fp8_block_scaled_mm_impl(a, b, a_scale, b_scale)

        def colleague_call(a_bf16=a_bf16, b=b, b_scale=b_scale):
            return impls["colleague_unquantized"](
                a_bf16, b, b_scale, [128, 128], torch.bfloat16
            )

        row = _compare_pair(
            ours_call,
            colleague_call,
            args.warmup,
            args.repeats,
            max(args.rtol, 0.02),
            max(args.atol, 0.125),
        )
        row.update(
            sig=case["sig"],
            M=case["M"],
            N=case["N"],
            K=case["K"],
            regime="decode_e2e",
            call_count=case["call_count"],
            colleague_route=(
                "fused_m1"
                if case["M"] == 1
                and impls["fused_configs"](case["N"], case["K"], 128, 128)
                else "quantize_plus_gemm_fallback"
            ),
        )
        rows.append(row)
        print(
            f"e2e   {index:02d}/{len(decode_cases)} {case['M']}x{case['N']}x{case['K']} "
            f"ours={row['ours_ms']:.5f}ms colleague={row['colleague_ms']:.5f}ms "
            f"speedup={row['ours_speedup']:.3f}x",
            flush=True,
        )
        del b, b_scale, a_bf16
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _run_intended_routes(impls, args) -> tuple[list[dict], list[dict]]:
    quantized_rows = []
    for n, k in TUNED_QUANTIZED_SHAPES:
        for m in (1, 2):
            a, b, a_scale, b_scale = _make_synthetic_inputs(m, n, k, n + k + m)

            def ours_call(a=a, b=b, a_scale=a_scale, b_scale=b_scale):
                return impls["ours"]._rdna4_fp8_block_scaled_mm_impl(
                    a, b, a_scale, b_scale
                )

            def colleague_call(a=a, b=b, a_scale=a_scale, b_scale=b_scale):
                return impls["colleague_quantized"](
                    a, b, a_scale, b_scale, [128, 128], torch.bfloat16
                )

            row = _compare_pair(
                ours_call,
                colleague_call,
                args.warmup,
                args.repeats,
                args.rtol,
                args.atol,
            )
            row.update(M=m, N=n, K=k, regime="intended_quantized")
            quantized_rows.append(row)
            print(
                f"tuned q {m}x{n}x{k} speedup={row['ours_speedup']:.3f}x",
                flush=True,
            )
            del a, b, a_scale, b_scale
            gc.collect()
            torch.cuda.empty_cache()

    fused_rows = []
    for n, k in TUNED_FUSED_SHAPES:
        _, b, _, b_scale = _make_synthetic_inputs(1, n, k, n + k + 101)
        gen = torch.Generator(device="cuda").manual_seed(n + k + 202)
        a_bf16 = torch.randn((1, k), device="cuda", dtype=torch.bfloat16, generator=gen)

        def ours_call(a_bf16=a_bf16, b=b, b_scale=b_scale):
            a, a_scale = impls["quantize"](
                a_bf16, 128, dtype=torch.float8_e4m3fn, use_ue8m0=False
            )
            return impls["ours"]._rdna4_fp8_block_scaled_mm_impl(a, b, a_scale, b_scale)

        def colleague_call(a_bf16=a_bf16, b=b, b_scale=b_scale):
            return impls["colleague_unquantized"](
                a_bf16, b, b_scale, [128, 128], torch.bfloat16
            )

        row = _compare_pair(
            ours_call,
            colleague_call,
            args.warmup,
            args.repeats,
            max(args.rtol, 0.02),
            max(args.atol, 0.125),
        )
        row.update(M=1, N=n, K=k, regime="intended_fused_e2e")
        fused_rows.append(row)
        print(f"tuned f 1x{n}x{k} speedup={row['ours_speedup']:.3f}x", flush=True)
        del b, b_scale, a_bf16
        gc.collect()
        torch.cuda.empty_cache()
    return quantized_rows, fused_rows


def main() -> None:
    global TIMING_GRAPH
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(HERE / "baseline" / "cases.json"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--output", default=str(HERE / "colleague_ab_results.json"))
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-intended", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--only-decode", action="store_true")
    args = parser.parse_args()

    if Path.cwd() != Path("/app/vllm-colleague"):
        raise SystemExit("run from /app/vllm-colleague with its PYTHONPATH")
    torch.set_grad_enabled(False)
    TIMING_GRAPH = args.graph
    impls = _load_implementations()
    cases = json.loads(Path(args.cases).read_text())["cases"]
    if args.only_decode:
        cases = [case for case in cases if case["regime"] == "decode"]
    exact = _run_exact_quantized(impls, cases, args)
    exact_e2e = [] if args.skip_e2e else _run_exact_decode_e2e(impls, cases, args)
    intended_quantized, intended_fused = ([], [])
    if not args.skip_intended:
        intended_quantized, intended_fused = _run_intended_routes(impls, args)

    result = {
        "schema": "geak.rdna4_colleague_ab.v1",
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cache_flush_mb": 512,
            "timing": (
                "cold-cache CUDA graph event, both orders"
                if args.graph
                else "cold-cache CUDA event, both orders"
            ),
            "our_commit": "fd381955ad9c402c0173eebaa42a738b091bd410",
            "colleague_commit": "e8370f66ad143a7fc8c5124bf3a98b9f1add52e4",
        },
        "exact_45_quantized": exact,
        "exact_45_summary": _summarize(exact),
        "exact_decode_e2e": exact_e2e,
        "exact_decode_e2e_summary": _summarize(exact_e2e) if exact_e2e else {},
        "intended_quantized": intended_quantized,
        "intended_quantized_summary": (
            _summarize(intended_quantized) if intended_quantized else {}
        ),
        "intended_fused_e2e": intended_fused,
        "intended_fused_e2e_summary": (
            _summarize(intended_fused) if intended_fused else {}
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exact_45": result["exact_45_summary"],
                "exact_decode_e2e": result["exact_decode_e2e_summary"],
                "intended_quantized": result["intended_quantized_summary"],
                "intended_fused_e2e": result["intended_fused_e2e_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
