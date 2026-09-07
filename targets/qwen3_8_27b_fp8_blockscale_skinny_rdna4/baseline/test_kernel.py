#!/usr/bin/env python3
"""Correctness and timing driver for the native-skinny frozen baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from kernel import run

HERE = Path(__file__).resolve().parent


def make_inputs(case: dict, draw: int = 0) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(int(case["seed"]) + draw * 100003)
    m, n, k = case["M"], case["N"], case["K"]
    # Row-dependent offsets make accidental row broadcast/mixing visible.
    rows = torch.arange(m, device="cuda", dtype=torch.float32)[:, None] * 0.03125
    a = (torch.randn((m, k), device="cuda", generator=gen) * 0.25 + rows).to(
        torch.float8_e4m3fn
    )
    logical_b = (torch.randn((n, k), device="cuda", generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = torch.empty_strided(
        (n, k), tuple(case["b_stride"]), device="cuda", dtype=torch.float8_e4m3fn
    )
    b.copy_(logical_b)
    k_blocks = k // 128
    a_scale = (
        torch.rand((m, k_blocks), device="cuda", generator=gen, dtype=torch.float32)
        * 0.05
        + 0.001
    )
    a_scale += torch.arange(m, device="cuda", dtype=torch.float32)[:, None] * 0.0001
    b_scale = (
        torch.rand(
            (n // 128, k_blocks), device="cuda", generator=gen, dtype=torch.float32
        )
        * 0.05
        + 0.001
    )
    return a, b, a_scale, b_scale


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
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        first = actual.clone()
        args[0].view(torch.uint8).bitwise_xor_(1)
        second = run(*args)
        if actual.data_ptr() == second.data_ptr():
            raise AssertionError(f"{case['sig']}: output storage was reused")
        if torch.equal(first, second):
            raise AssertionError(f"{case['sig']}: output did not reflect changed input")


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
    parser.add_argument("--case")
    parser.add_argument("--m", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--benchmark-json")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())["cases"]
    if args.case:
        cases = [case for case in cases if case["sig"] == args.case]
    if args.m is not None:
        cases = [case for case in cases if case["M"] == args.m]
    if not cases:
        raise SystemExit("no matching cases")
    if args.list:
        for case in cases:
            print(case["sig"])
        return

    results = {}
    for case in cases:
        validate(case, args.draws)
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
