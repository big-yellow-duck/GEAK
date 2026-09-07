#!/usr/bin/env python3
"""Correctness and allocation-free timing for broad RDNA4 SplitKV coverage."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from pathlib import Path

FLYDSL_SEED_ROOT = Path(os.environ.get("FLYDSL_SEED_ROOT", "/app/FlyDSL")).resolve()
for path in (FLYDSL_SEED_ROOT / "build-fly/python_packages", FLYDSL_SEED_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# Load FlyDSL's compiler libraries before torch loads ROCm's LLVM stack.
import flydsl  # noqa: F401
import torch
from kernel import route_for, run, run_triton

HERE = Path(__file__).resolve().parent
RTOL = 0.01
ATOL = 0.01


def _load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["cases"]


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp8": torch.float8_e4m3fn,
        "fp8fnuz": torch.float8_e4m3fnuz,
    }[name]


def _fill_cache(
    tensor: torch.Tensor,
    generator: torch.Generator,
    magnitude: float,
    quant_scale: float,
) -> None:
    temporary = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(magnitude)
    if tensor.element_size() == 1:
        temporary.div_(quant_scale)
    tensor.copy_(temporary.to(tensor.dtype))


def make_inputs(case: dict, draw: int = 0) -> tuple:
    seed = int(case["seed"]) + draw * 100_003
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seq_values = [int(value) for value in case["seq_lens"]]
    batch = len(seq_values)
    head_size = int(case["head_size"])
    num_query_heads = int(case["num_query_heads"])
    num_kv_heads = int(case["num_kv_heads"])
    page_size = int(case["page_size"])
    query_dtype = _dtype(case["query_dtype"])
    kv_dtype = _dtype(case["kv_dtype"])
    input_scale = float(case["input_scale"])
    k_scale_value = float(case["k_scale"])
    v_scale_value = float(case["v_scale"])

    query_f32 = torch.randn(
        (batch, num_query_heads, head_size),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).mul_(input_scale)
    # Make query rows distinct so a padded-GQA implementation cannot silently
    # broadcast or alias heads while still passing random tests by chance.
    query_f32.add_(
        torch.arange(num_query_heads, device="cuda", dtype=torch.float32)[None, :, None]
        * 0.00390625
    )
    query = query_f32.to(query_dtype)

    pages_per_seq = [math.ceil(length / page_size) for length in seq_values]
    num_blocks = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    pack = 16 // kv_dtype.itemsize
    key_shape = (num_blocks, num_kv_heads, head_size // pack, page_size, pack)
    value_shape = (num_blocks, num_kv_heads, head_size, page_size)
    block_span = num_kv_heads * head_size * page_size
    key_stride = (
        block_span * 2,
        head_size * page_size,
        page_size * pack,
        pack,
        1,
    )
    value_stride = (
        block_span * 2,
        head_size * page_size,
        page_size,
        1,
    )
    key_cache = torch.empty_strided(
        key_shape, key_stride, dtype=kv_dtype, device="cuda"
    )
    value_cache = torch.empty_strided(
        value_shape, value_stride, dtype=kv_dtype, device="cuda"
    )
    _fill_cache(key_cache, generator, input_scale, k_scale_value)
    _fill_cache(value_cache, generator, input_scale, v_scale_value)

    block_tables = torch.zeros((batch, max_pages), device="cuda", dtype=torch.int32)
    physical_order = torch.randperm(num_blocks, generator=generator, device="cuda")
    offset = 0
    for seq_index, pages in enumerate(pages_per_seq):
        block_tables[seq_index, :pages] = physical_order[offset : offset + pages]
        offset += pages
    seq_lens = torch.tensor(seq_values, device="cuda", dtype=torch.int32)
    query_start_loc = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    k_scale = torch.tensor(k_scale_value, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(v_scale_value, device="cuda", dtype=torch.float32)
    output = torch.empty_like(query)
    splits = int(case["splits"])
    mid_out = torch.empty(
        (batch, num_query_heads, splits, head_size),
        device="cuda",
        dtype=torch.float32,
    )
    mid_lse = torch.empty(
        (batch, num_query_heads, splits), device="cuda", dtype=torch.float32
    )
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
        mid_out,
        mid_lse,
        splits,
        float(case["scale"]),
    )


def _with_fresh_workspace(inputs: tuple) -> tuple:
    query = inputs[0]
    splits = inputs[11]
    return (
        *inputs[:8],
        torch.empty_like(query),
        torch.empty(
            (query.shape[0], query.shape[1], splits, query.shape[2]),
            device=query.device,
            dtype=torch.float32,
        ),
        torch.empty(
            (query.shape[0], query.shape[1], splits),
            device=query.device,
            dtype=torch.float32,
        ),
        *inputs[11:],
    )


def validate(case: dict, draws: int) -> dict:
    worst_abs = 0.0
    worst_rel = 0.0
    for draw in range(draws):
        inputs = make_inputs(case, draw)
        reference_inputs = _with_fresh_workspace(inputs)
        actual_inputs = _with_fresh_workspace(inputs)
        expected = run_triton(*reference_inputs).clone()
        actual = run(*actual_inputs).clone()
        torch.cuda.synchronize()
        delta = (actual.float() - expected.float()).abs()
        denominator = expected.float().abs().clamp_min(ATOL)
        worst_abs = max(worst_abs, float(delta.max()))
        worst_rel = max(worst_rel, float((delta / denominator).max()))
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)

    independent_a = make_inputs(case, draws + 17)
    independent_b = make_inputs(case, draws + 29)
    output_a = run(*independent_a)
    held_a = output_a.clone()
    output_b = run(*independent_b)
    if output_a.data_ptr() == output_b.data_ptr():
        raise AssertionError(f"{case['name']}: output storage was reused")
    if torch.equal(held_a, output_b):
        raise AssertionError(f"{case['name']}: output ignored changed inputs")
    return {"max_abs": worst_abs, "max_rel": worst_rel}


def time_case(case: dict, warmup: int, repeats: int, launches: int) -> dict:
    inputs = make_inputs(case)
    for _ in range(warmup):
        run(*inputs)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launches):
            run(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / launches)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
        "launches_per_sample": launches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--role", action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--benchmark-json")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.case:
        wanted = set(args.case)
        cases = [item for item in cases if item["name"] in wanted]
    if args.role:
        wanted_roles = set(args.role)
        cases = [item for item in cases if item["coverage_role"] in wanted_roles]
    if not cases:
        raise SystemExit("no matching cases")
    if args.list:
        for item in cases:
            print(
                f"{item['name']} role={item['coverage_role']} weight={item['weight']}"
            )
        return

    results = {}
    for case_index, item in enumerate(cases):
        correctness = validate(item, args.draws)
        inputs = make_inputs(item)
        route = route_for(inputs[0], inputs[1], inputs[4])
        record = {
            "route": route,
            "correctness": correctness,
            "weight": item["weight"],
        }
        if not args.correctness_only:
            record.update(time_case(item, args.warmup, args.repeats, args.launches))
        results[item["name"]] = record
        suffix = "" if args.correctness_only else f" {record['median_ms']:.6f} ms"
        print(
            f"{item['name']}: PASS route={route} "
            f"max_abs={correctness['max_abs']:.6g}{suffix}",
            flush=True,
        )
        if case_index + 1 != len(cases):
            gc.collect()
            torch.cuda.empty_cache()

    if args.benchmark_json:
        Path(args.benchmark_json).write_text(
            json.dumps(
                {"schema": "geak.rdna4_splitkv_timings.v1", "cases": results},
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
