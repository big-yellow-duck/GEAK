#!/usr/bin/env python3
"""Correctness and allocation-free timing harness for FP8/BF16 SplitKV."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from kernel import (
    COMPUTE_BLOCK_SIZE,
    HEAD_SIZE,
    MAX_NUM_SPLITS,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    PHYSICAL_BLOCK_SIZE,
    choose_num_splits,
    run,
)

HERE = Path(__file__).resolve().parent
FP8_KEY_STRIDE = (1605632, 401408, 25088, 16, 1)
BF16_KEY_STRIDE = (1605632, 401408, 12544, 8, 1)
VALUE_STRIDE = (1605632, 401408, 1568, 1)


def _load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["cases"]


def _fill_fp8(tensor: torch.Tensor, generator: torch.Generator) -> None:
    # Generate finite values in BF16, then quantize into the exact strided FP8 view.
    temporary = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.25)
    tensor.copy_(temporary)
    del temporary


def make_inputs(case: dict, seed: int) -> tuple[torch.Tensor, ...]:
    seq_values = [int(value) for value in case["seq_lens"]]
    batch = len(seq_values)
    pages_per_seq = [math.ceil(value / PHYSICAL_BLOCK_SIZE) for value in seq_values]
    num_blocks = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    query = torch.randn(
        (batch, NUM_QUERY_HEADS, HEAD_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.25)
    kv_kind = case.get("kv_dtype", "fp8")
    if kv_kind == "fp8":
        kv_dtype = torch.float8_e4m3fn
        x = 16
        key_stride = FP8_KEY_STRIDE
        k_scale_value, v_scale_value = 0.75, 1.25
    elif kv_kind == "bf16":
        kv_dtype = torch.bfloat16
        x = 8
        key_stride = BF16_KEY_STRIDE
        k_scale_value = v_scale_value = 1.0
    else:
        raise ValueError(f"unsupported kv_dtype: {kv_kind}")
    key_cache = torch.empty_strided(
        (num_blocks, NUM_KV_HEADS, HEAD_SIZE // x, PHYSICAL_BLOCK_SIZE, x),
        key_stride,
        device="cuda",
        dtype=kv_dtype,
    )
    value_cache = torch.empty_strided(
        (num_blocks, NUM_KV_HEADS, HEAD_SIZE, PHYSICAL_BLOCK_SIZE),
        VALUE_STRIDE,
        device="cuda",
        dtype=kv_dtype,
    )
    _fill_fp8(key_cache, generator)
    _fill_fp8(value_cache, generator)

    block_tables = torch.zeros((batch, max_pages), device="cuda", dtype=torch.int32)
    next_block = 0
    for seq_idx, pages in enumerate(pages_per_seq):
        block_tables[seq_idx, :pages] = torch.arange(
            next_block, next_block + pages, device="cuda", dtype=torch.int32
        )
        next_block += pages
    seq_lens = torch.tensor(seq_values, device="cuda", dtype=torch.int32)
    k_scale = torch.tensor(k_scale_value, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(v_scale_value, device="cuda", dtype=torch.float32)
    output = torch.empty_like(query)
    mid_out = torch.empty(
        (batch, NUM_QUERY_HEADS, MAX_NUM_SPLITS, HEAD_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    mid_lse = torch.empty(
        (batch, NUM_QUERY_HEADS, MAX_NUM_SPLITS),
        device="cuda",
        dtype=torch.float32,
    )
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        output,
        mid_out,
        mid_lse,
    )


def incumbent_output(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Run vLLM's stride-aware 2D FP8 kernel as an independent seed check."""
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        kernel_paged_attention_2d,
    )

    query, key_cache, value_cache, block_tables, seq_lens, k_scale, v_scale, *_ = inputs
    batch = query.shape[0]
    output = torch.empty_like(query)
    kernel_paged_attention_2d[(batch, NUM_KV_HEADS)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=1.0 / math.sqrt(HEAD_SIZE),
        k_scale=k_scale,
        v_scale=v_scale,
        out_scale_inv=1.0,
        num_query_heads=NUM_QUERY_HEADS,
        num_queries_per_kv=NUM_QUERY_HEADS // NUM_KV_HEADS,
        num_queries_per_kv_padded=16,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=COMPUTE_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=PHYSICAL_BLOCK_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE,
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[-1],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=False,
        query_start_len_ptr=None,
        USE_SINKS=False,
        USE_FP8=False,
    )
    return output


def check_case(case: dict, inputs: tuple[torch.Tensor, ...]) -> dict:
    expected = incumbent_output(inputs)
    actual = run(*inputs).clone()
    torch.cuda.synchronize()
    delta = (actual.float() - expected.float()).abs()
    if not torch.allclose(actual.float(), expected.float(), rtol=0.01, atol=0.01):
        raise AssertionError(
            f"{case['name']}: SplitKV mismatch: max_abs={delta.max().item():.6g}, "
            f"mean_abs={delta.mean().item():.6g}"
        )
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
    }


def time_case(inputs: tuple[torch.Tensor, ...], warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        run(*inputs)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
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
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--benchmark-json")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case["name"] in requested]
        if not cases:
            raise SystemExit(f"unknown case(s): {', '.join(args.case)}")
    if args.list:
        for case in cases:
            print(case["name"])
        return

    results = {}
    for case_idx, case in enumerate(cases):
        inputs = make_inputs(case, seed=20260824 + case_idx)
        seq_lens = case["seq_lens"]
        splits = choose_num_splits(len(seq_lens), max(seq_lens))
        record = {
            "batch": len(seq_lens),
            "max_seq_len": max(seq_lens),
            "kv_dtype": case.get("kv_dtype", "fp8"),
            "num_splits": splits,
            "count": case["count"],
            "weight": case["weight"],
        }
        if args.check:
            record["correctness"] = check_case(case, inputs)
        record.update(time_case(inputs, args.warmup, args.repeats))
        results[case["name"]] = record
        correctness = " checked" if args.check else ""
        print(
            f"{case['name']}: {record['median_ms']:.6f} ms, "
            f"splits={splits}{correctness}"
        )
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    if args.benchmark_json:
        Path(args.benchmark_json).write_text(
            json.dumps(
                {"schema": "geak.fp8_splitkv_baseline_timings.v1", "cases": results},
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
