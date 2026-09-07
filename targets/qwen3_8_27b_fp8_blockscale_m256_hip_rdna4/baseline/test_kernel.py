#!/usr/bin/env python3
"""Correctness and timing driver for the M256 HIP-vs-Triton campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from kernel import run

HERE = Path(__file__).resolve().parent
RTOL = 0.02
ATOL = 0.0625


def dtype_of(name: str) -> torch.dtype:
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


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
    generator: torch.Generator,
) -> torch.Tensor:
    dtype = dtype_of(dtype_name)
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
        value = (
            torch.randn(shape, device="cuda", generator=generator, dtype=dtype) * 0.25
        )
    if stride == contiguous_stride(shape):
        return value
    result = torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
    result.copy_(value)
    return result


def make_inputs(case: dict, draw: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(
        int(case["seed"]) + draw * 100_003
    )
    tensors = [
        make_tensor(
            case[f"{prefix}_shape"],
            case[f"{prefix}_stride"],
            case[f"{prefix}_dtype"],
            generator,
        )
        for prefix in ("a", "b", "as", "bs")
    ]
    # Distinct activation rows expose row broadcast and row-mixing bugs.
    rows = torch.arange(case["M"], device="cuda", dtype=torch.float32)[:, None]
    tensors[0].copy_((tensors[0].float() + rows * 0.03125).to(torch.float8_e4m3fn))
    tensors[2].add_(rows * 0.0001)
    return tuple(tensors)


def reference(args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(*args, [128, 128], torch.bfloat16)


def validate(case: dict, draws: int) -> None:
    for draw in range(draws):
        args = make_inputs(case, draw)
        expected = reference(args)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)

        held = actual.clone()
        args[0].view(torch.uint8).bitwise_xor_(1)
        changed = run(*args)
        if actual.data_ptr() == changed.data_ptr():
            raise AssertionError(f"{case['sig']}: output storage was reused")
        if torch.equal(held, changed):
            raise AssertionError(
                f"{case['sig']}: output did not reflect changed activation"
            )


def time_case(case: dict, warmup: int, repeats: int) -> dict:
    args = make_inputs(case)
    for _ in range(warmup):
        run(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if tuple(out.shape) != (case["M"], case["N"]):
        raise AssertionError(f"{case['sig']}: wrong output shape {tuple(out.shape)}")
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--m", action="append", type=int)
    parser.add_argument("--role", action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--benchmark-json")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())["cases"]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["sig"] in wanted]
    if args.m:
        wanted_m = set(args.m)
        cases = [case for case in cases if case["M"] in wanted_m]
    if args.role:
        wanted_roles = set(args.role)
        cases = [case for case in cases if case["coverage_role"] in wanted_roles]
    if not cases:
        raise SystemExit("no matching cases")
    if args.list:
        for case in cases:
            print(
                f"{case['sig']} role={case['coverage_role']} "
                f"route={case['route_expectation']} weight={case['weight']:.6f}"
            )
        return

    results = {}
    for case in cases:
        validate(case, args.draws)
        if args.correctness_only:
            print(f"{case['sig']}: PASS")
            continue
        result = time_case(case, args.warmup, args.repeats)
        results[case["sig"]] = result
        print(f"{case['sig']}: {result['median_ms']:.6f} ms")
    if args.benchmark_json:
        Path(args.benchmark_json).write_text(
            json.dumps(
                {"schema": "geak.baseline_timings.v1", "cases": results}, indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
