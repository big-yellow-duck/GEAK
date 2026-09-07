"""Hybrid seed for replacing vLLM's RDNA4 Triton SplitKV fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

FLYDSL_SEED_ROOT = Path(os.environ.get("FLYDSL_SEED_ROOT", "/app/FlyDSL")).resolve()
VLLM_SRC = Path(
    os.environ.get("VLLM_SRC", "/app/vllm-rdna4-fp8-flydsl-tp2-hip-ar")
).resolve()
for path in (
    FLYDSL_SEED_ROOT / "build-fly/python_packages",
    FLYDSL_SEED_ROOT,
    VLLM_SRC,
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# FlyDSL must load before torch brings the ROCm LLVM stack into the process.
import flydsl  # noqa: F401
import torch
from kernels.attention.rdna4_splitkv import (
    rdna4_splitkv_paged_attention,
)
from vllm.v1.attention.ops import (
    chunked_prefill_paged_decode as paged_decode,
)

# This target measures replacement of Triton itself. Prevent the development
# vLLM checkout from recursively selecting its HIP or FlyDSL experimental path.
paged_decode._can_use_rdna4_splitkv_paged_attention = lambda **_: False
paged_decode._can_use_rdna4_flydsl_splitkv_paged_attention = lambda **_: False


def route_for(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
) -> str:
    """Return the initial route; GEAK should shrink the Triton region."""

    num_kv_heads = int(key_cache.shape[1])
    gqa = int(query.shape[1]) // num_kv_heads
    seed_flydsl = (
        query.dtype == torch.bfloat16
        and key_cache.dtype == torch.float8_e4m3fn
        and query.shape[2] == 256
        and seq_lens.numel() == 1
        and gqa in (6, 7)
        and key_cache.shape[3] >= 8
        and key_cache.shape[3] % 8 == 0
    )
    return "flydsl_seed" if seed_flydsl else "vllm_triton"


def run_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Run the immutable Triton SplitKV reference with caller-owned storage."""

    return paged_decode._paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=splits,
        mid_out=mid_out,
        mid_lse=mid_lse,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )


def run(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Run the current FlyDSL winner or the live Triton fallback."""

    if route_for(query, key_cache, seq_lens) == "flydsl_seed":
        rdna4_splitkv_paged_attention(
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
            scale=scale,
            validate=False,
        )
        return output

    return run_triton(
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
        scale,
    )


__all__ = ["route_for", "run", "run_triton"]
