#!/usr/bin/env python3
"""Reproducible correctness/timing driver for the frozen Triton baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from kernel import run

HERE = Path(__file__).resolve().parent


def dtype_of(name: str) -> torch.dtype:
    table = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return table[name]


def contiguous_stride(shape: list[int]) -> list[int]:
    stride = 1
    result = []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def make_tensor(
    shape: list[int],
    stride: list[int],
    dtype_name: str,
    gen: torch.Generator,
) -> torch.Tensor:
    dtype = dtype_of(dtype_name)
    if dtype == torch.float8_e8m0fnu:
        raw = torch.randint(122, 130, shape, dtype=torch.uint8, device="cuda", generator=gen)
        value = raw.view(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        value = (torch.randn(shape, device="cuda", generator=gen) * 0.25).to(dtype)
    elif dtype == torch.float32:
        value = torch.rand(shape, device="cuda", generator=gen, dtype=dtype) * 0.05 + 0.001
    else:
        value = torch.randn(shape, device="cuda", generator=gen, dtype=dtype) * 0.25

    expected_contiguous = contiguous_stride(shape)
    if stride == expected_contiguous:
        return value
    result = torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
    result.copy_(value)
    return result


def make_inputs(case: dict) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(int(case["seed"]))
    return (
        make_tensor(case["a_shape"], case["a_stride"], case["a_dtype"], gen),
        make_tensor(case["b_shape"], case["b_stride"], case["b_dtype"], gen),
        make_tensor(case["as_shape"], case["as_stride"], case["as_dtype"], gen),
        make_tensor(case["bs_shape"], case["bs_stride"], case["bs_dtype"], gen),
    )


def time_case(case: dict, warmup: int, repeats: int) -> dict:
    args = make_inputs(case)
    out_dtype = dtype_of(case["output_dtype"])
    for _ in range(warmup):
        run(*args, block_size=case["block_size"], output_dtype=out_dtype)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run(*args, block_size=case["block_size"], output_dtype=out_dtype)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if tuple(out.shape) != (case["M"], case["N"]):
        raise AssertionError(f"{case['sig']}: wrong output shape {tuple(out.shape)}")
    if not torch.isfinite(out.float()).all():
        raise AssertionError(f"{case['sig']}: non-finite output")
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(HERE / "cases.json"))
    p.add_argument("--case")
    p.add_argument("--list", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--benchmark-json")
    args = p.parse_args()
    cases = json.loads(Path(args.cases).read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["sig"] == args.case]
        if not cases:
            raise SystemExit(f"unknown case: {args.case}")
    if args.list:
        for c in cases:
            print(c["sig"])
        return
    results = {}
    for case in cases:
        result = time_case(case, args.warmup, args.repeats)
        results[case["sig"]] = result
        print(f"{case['sig']}: {result['median_ms']:.6f} ms")
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(
            json.dumps({"schema": "geak.baseline_timings.v1", "cases": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
