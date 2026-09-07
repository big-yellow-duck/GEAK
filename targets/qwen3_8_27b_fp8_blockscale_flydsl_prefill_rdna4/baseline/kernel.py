"""FlyDSL M<=64 seed with the live vLLM prefill fallback as denominator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

FLYDSL_SEED_ROOT = Path(
    os.environ.get("FLYDSL_SEED_ROOT", "/app/rdna4_fp8_blockscale_flydsl")
).resolve()
for path in (FLYDSL_SEED_ROOT / "build-fly/python_packages", FLYDSL_SEED_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# FlyDSL must load its compiler libraries before torch loads HIP's LLVM stack.
import flydsl  # noqa: F401
import torch
from kernels.gemm.rdna4_fp8_blockscale import rdna4_fp8_block_scaled_mm

VLLM_SRC = Path(os.environ.get("VLLM_SRC", "/app/vllm")).resolve()
if str(VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLLM_SRC))


def route_for(m: int) -> str:
    """Return the seed backend; accepted work should expand the FlyDSL interval."""
    return "flydsl" if 1 <= m <= 64 else "vllm_triton"


def run(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | list[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the current FlyDSL routes and retain a correct broad-prefill fallback."""
    if tuple(block_size) != (128, 128):
        raise ValueError(f"expected block_size=(128, 128), got {tuple(block_size)}")
    if output_dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 output, got {output_dtype}")

    m = int(a.shape[0])
    if route_for(m) == "flydsl":
        return rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale)

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
