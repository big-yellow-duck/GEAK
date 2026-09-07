"""FP8/BF16 paged-decode SplitKV oracle for Qwen3.8-27B on RDNA4.

Derived from vLLM PR #45916 plus feiyehua/vllm PR #2.  The seed deliberately
keeps allocation outside the timed callable and adds the missing FP8 KV scales.
"""

from __future__ import annotations

import math

import torch
from vllm.triton_utils import tl, triton

PHYSICAL_BLOCK_SIZE = 1568
HEAD_SIZE = 256
NUM_QUERY_HEADS = 12
NUM_KV_HEADS = 2
MAX_NUM_SPLITS = 16
COMPUTE_BLOCK_SIZE = 32


@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _splitkv_stage1(
    mid_out,
    mid_lse,
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
    sm_scale,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    stride_k_0: tl.int64,
    stride_k_1: tl.int64,
    stride_k_2: tl.int64,
    stride_k_3: tl.int64,
    stride_k_4: tl.int64,
    stride_v_0: tl.int64,
    stride_v_1: tl.int64,
    stride_v_2: tl.int64,
    stride_v_3: tl.int64,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_QUERIES_PER_KV_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE_CONST: tl.constexpr,
    HEAD_SIZE_CONST: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    X: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens + seq_idx)
    num_splits = tl.num_programs(2)
    split_len = _cdiv(_cdiv(seq_len, num_splits), BLOCK_SIZE) * BLOCK_SIZE
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, seq_len)

    q_head = kv_head_idx * NUM_QUERIES_PER_KV + tl.arange(0, NUM_QUERIES_PER_KV_PADDED)
    head_mask = q_head < (kv_head_idx + 1) * NUM_QUERIES_PER_KV
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE_CONST

    q_offset = seq_idx * query_stride_0 + q_head[:, None] * query_stride_1
    q = tl.load(
        query + q_offset + offs_d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    running_max = tl.full([NUM_QUERIES_PER_KV_PADDED], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([NUM_QUERIES_PER_KV_PADDED], dtype=tl.float32)
    acc = tl.zeros([NUM_QUERIES_PER_KV_PADDED, HEAD_SIZE_PADDED], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE)
    table_base = seq_idx * block_table_stride

    for start_n in tl.range(split_start, split_end, BLOCK_SIZE):
        token = start_n + offs_n
        logical_block = token // PHYSICAL_BLOCK_SIZE_CONST
        physical_block = tl.load(block_tables + table_base + logical_block)
        in_block = token % PHYSICAL_BLOCK_SIZE_CONST
        token_mask = token < split_end

        k_offset = (
            physical_block[None, :] * stride_k_0
            + kv_head_idx * stride_k_1
            + (offs_d[:, None] // X) * stride_k_2
            + in_block[None, :] * stride_k_3
            + (offs_d[:, None] % X) * stride_k_4
        )
        k_fp8 = tl.load(
            key_cache + k_offset,
            mask=dim_mask[:, None] & token_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        # FP8 cache values are scaled in FP32, then consumed by BF16 dots.
        k = (k_fp8.to(tl.float32) * tl.load(k_scale)).to(q.dtype)

        v_offset = (
            physical_block[:, None] * stride_v_0
            + kv_head_idx * stride_v_1
            + offs_d[None, :] * stride_v_2
            + in_block[:, None] * stride_v_3
        )
        v_fp8 = tl.load(
            value_cache + v_offset,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        v = (v_fp8.to(tl.float32) * tl.load(v_scale)).to(q.dtype)

        scores = sm_scale * tl.dot(q, k)
        scores = tl.where(
            head_mask[:, None] & token_mask[None, :], scores, float("-inf")
        )
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        p = tl.exp(scores - new_max[:, None])
        p = tl.where(new_max[:, None] == float("-inf"), 0.0, p)
        block_sum = tl.sum(p, axis=1)
        alpha = tl.exp(running_max - new_max)
        alpha = tl.where(running_max == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        running_sum = running_sum * alpha + block_sum
        running_max = new_max

    out_offset = (
        seq_idx * mid_out_stride_0
        + q_head[:, None] * mid_out_stride_1
        + split_idx * mid_out_stride_2
        + offs_d[None, :]
    )
    lse_offset = (
        seq_idx * mid_lse_stride_0
        + q_head * mid_lse_stride_1
        + split_idx * mid_lse_stride_2
    )
    has_tokens = split_end > split_start
    tl.store(
        mid_out + out_offset,
        acc / (running_sum[:, None] + 1e-10),
        mask=has_tokens & head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        mid_lse + lse_offset,
        running_max + tl.log(running_sum),
        mask=has_tokens & head_mask,
    )


@triton.jit
def _splitkv_reduce(
    output,
    mid_out,
    mid_lse,
    seq_lens,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    HEAD_SIZE_CONST: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    seq_len = tl.load(seq_lens + seq_idx)
    split_len = _cdiv(_cdiv(seq_len, NUM_SPLITS), BLOCK_SIZE) * BLOCK_SIZE
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE_CONST

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)
    for split_idx in tl.range(0, NUM_SPLITS, num_stages=2):
        split_start = split_idx * split_len
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            lse = tl.load(
                mid_lse
                + seq_idx * mid_lse_stride_0
                + q_head * mid_lse_stride_1
                + split_idx * mid_lse_stride_2
            )
            partial = tl.load(
                mid_out
                + seq_idx * mid_out_stride_0
                + q_head * mid_out_stride_1
                + split_idx * mid_out_stride_2
                + offs_d,
                mask=dim_mask,
                other=0.0,
            )
            new_max = tl.maximum(running_max, lse)
            alpha = tl.exp(running_max - new_max)
            beta = tl.exp(lse - new_max)
            acc = acc * alpha + partial * beta
            running_sum = running_sum * alpha + beta
            running_max = new_max

    tl.store(
        output + seq_idx * output_stride_0 + q_head * output_stride_1 + offs_d,
        acc / (running_sum + 1e-10),
        mask=dim_mask,
    )


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def choose_num_splits(
    batch_size: int, max_seq_len: int, num_sms: int | None = None
) -> int:
    """The upstream occupancy seed; GEAK is expected to improve this policy."""
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch_nheads = batch_size * NUM_KV_HEADS
    num_n_blocks = _ceil_div(max_seq_len, COMPUTE_BLOCK_SIZE)
    target_workgroups = 2 * num_sms
    if batch_nheads >= 0.8 * target_workgroups or num_n_blocks < 2 * num_sms:
        return 1
    max_splits = min(MAX_NUM_SPLITS, num_sms, num_n_blocks)
    efficiencies: list[float] = []
    maximum = 0.0
    for splits in range(1, max_splits + 1):
        eligible = splits == 1 or _ceil_div(num_n_blocks, splits) != _ceil_div(
            num_n_blocks, splits - 1
        )
        if not eligible:
            efficiencies.append(0.0)
            continue
        waves = batch_nheads * splits / target_workgroups
        efficiency = waves / math.ceil(waves)
        efficiencies.append(efficiency)
        maximum = max(maximum, efficiency)
    for splits, efficiency in enumerate(efficiencies, start=1):
        if efficiency >= 0.85 * maximum:
            return splits
    return 1


def run(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
) -> torch.Tensor:
    """Run complete stage-1 plus reduction with caller-owned scratch buffers."""
    batch_size, num_query_heads, head_size = query.shape
    if (num_query_heads, head_size) != (NUM_QUERY_HEADS, HEAD_SIZE):
        raise ValueError(
            f"expected Qwen TP2 query [B,12,256], got {tuple(query.shape)}"
        )
    if key_cache.dtype != value_cache.dtype:
        raise TypeError("key/value cache dtypes must match")
    if key_cache.dtype == torch.float8_e4m3fn:
        x = 16
    elif key_cache.dtype == torch.bfloat16:
        x = 8
    else:
        raise TypeError("this target requires FP8 E4M3FN or BF16 KV caches")
    if key_cache.shape[1:] != (
        NUM_KV_HEADS,
        HEAD_SIZE // x,
        PHYSICAL_BLOCK_SIZE,
        x,
    ):
        raise ValueError(f"unexpected key-cache geometry: {tuple(key_cache.shape)}")
    if value_cache.shape[1:] != (NUM_KV_HEADS, HEAD_SIZE, PHYSICAL_BLOCK_SIZE):
        raise ValueError(f"unexpected value-cache geometry: {tuple(value_cache.shape)}")

    num_splits = choose_num_splits(batch_size, int(seq_lens.max().item()))
    q_per_kv = NUM_QUERY_HEADS // NUM_KV_HEADS
    q_per_kv_padded = max(triton.next_power_of_2(q_per_kv), 16)
    sm_scale = 1.0 / math.sqrt(HEAD_SIZE)

    _splitkv_stage1[(batch_size, NUM_KV_HEADS, num_splits)](
        mid_out,
        mid_lse,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        sm_scale,
        block_tables.stride(0),
        query.stride(0),
        query.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.stride(4),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        NUM_QUERIES_PER_KV=q_per_kv,
        NUM_QUERIES_PER_KV_PADDED=q_per_kv_padded,
        BLOCK_SIZE=COMPUTE_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE_CONST=PHYSICAL_BLOCK_SIZE,
        HEAD_SIZE_CONST=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE,
        X=x,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    _splitkv_reduce[(batch_size, NUM_QUERY_HEADS)](
        output,
        mid_out,
        mid_lse,
        seq_lens,
        output.stride(0),
        output.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        HEAD_SIZE_CONST=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE,
        NUM_SPLITS=num_splits,
        BLOCK_SIZE=COMPUTE_BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output
