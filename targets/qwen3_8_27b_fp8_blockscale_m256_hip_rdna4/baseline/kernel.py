"""Frozen Triton denominator for the RDNA4 M256 native-HIP campaign."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

VLLM_SRC = Path(os.environ.get("VLLM_SRC", "/app/vllm")).resolve()
if str(VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLLM_SRC))


def route_for(
    _a: torch.Tensor,
    _weight: torch.Tensor,
    _a_scale: torch.Tensor,
    _weight_scale: torch.Tensor,
) -> str:
    """Name the frozen denominator; accepted candidates must replace this route."""
    return "vllm_triton"


def run(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | list[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Execute vLLM's live Triton block-scaled FP8 GEMM as the oracle baseline."""
    if tuple(block_size) != (128, 128):
        raise ValueError(f"expected block_size=(128, 128), got {tuple(block_size)}")
    if output_dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 output, got {output_dtype}")

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


__all__ = ["route_for", "run"]
