"""Frozen-candidate entry point for vLLM's Triton block-FP8 GEMM."""

from __future__ import annotations

import os
import sys

import torch

VLLM_SRC = os.environ.get("VLLM_SRC", "/app/vllm")
if VLLM_SRC not in sys.path:
    sys.path.insert(0, VLLM_SRC)


def run(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | list[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(
        a,
        weight,
        a_scale,
        weight_scale,
        list(block_size),
        output_dtype,
    )
