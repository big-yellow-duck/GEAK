"""Frozen gfx12x hybrid baseline for the M=1..16 native expansion."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

VLLM_SRC = os.environ.get("VLLM_SRC", "/app/vllm")
if VLLM_SRC not in sys.path:
    sys.path.insert(0, VLLM_SRC)

_decode_ext = None


def _get_decode_ext():
    global _decode_ext
    if _decode_ext is None:
        from torch.utils.cpp_extension import load_inline

        source = Path(__file__).with_name("binding.cpp").read_text()
        _decode_ext = load_inline(
            name="geak_rdna4_skinny_m1_16_baseline",
            cpp_sources="",
            cuda_sources=source,
            functions=None,
            extra_cuda_cflags=[
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-O3",
            ],
            with_cuda=True,
            verbose=False,
        )
    return _decode_ext


def run(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | list[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    m, k = a.shape
    n = weight.shape[0]
    supported = (
        1 <= m <= 16
        and n % 128 == 0
        and k % 128 == 0
        and tuple(block_size) == (128, 128)
        and output_dtype == torch.bfloat16
        and a.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and a_scale.dtype == torch.float32
        and weight_scale.dtype == torch.float32
        and a.is_contiguous()
        and weight.stride(1) == 1
        and a_scale.is_contiguous()
        and weight_scale.is_contiguous()
    )
    if supported and m in (1, 2):
        return _get_decode_ext().decode(a, weight, a_scale, weight_scale)

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
