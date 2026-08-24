"""Frozen-candidate entry point for vLLM's Triton block-FP8 GEMM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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
        name="geak_rdna4_decode_k128",
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


@triton.jit
def _prefill_bm32_bn128(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    stride_am,
    stride_bm,
    stride_asm,
    stride_ask,
    stride_bsm,
    stride_bsk,
    GROUP_M: tl.constexpr,
):
    """One 32x128 output tile with an exact scaled partial per 128 K."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 32)
    num_pid_n = tl.cdiv(N, 128)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * 32 + tl.arange(0, 32)) % M
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * stride_bm + offs_k[:, None]
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + pid_n * stride_bsm

    accumulator = tl.zeros((32, 128), dtype=tl.float32)
    for k_block in range(tl.cdiv(K, 128)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + k_block * stride_ask)
        b_s = tl.load(bs_ptrs + k_block * stride_bsk)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 128
        b_ptrs += 128

    out_m = pid_m * 32 + tl.arange(0, 32)
    out_n = pid_n * 128 + tl.arange(0, 128)
    tl.store(
        C + out_m[:, None] * N + out_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=out_m[:, None] < M,
    )


def run(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int] | list[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M, K = a.shape
    N = weight.shape[0]
    if (
        M in (1, 2)
        and K % 128 == 0
        and N % 128 == 0
        and block_size[0] == 128
        and block_size[1] == 128
        and output_dtype == torch.bfloat16
        and torch.cuda.get_device_capability(a.device) == (12, 0)
    ):
        return _get_decode_ext().decode(a, weight, a_scale, weight_scale)

    use_bm32 = ((M == 523 and N != 17408)
                or (M == 784 and N not in (7168, 17408)))
    if use_bm32:
        out = torch.empty((M, N), device=a.device, dtype=output_dtype)
        grid = (triton.cdiv(M, 32) * triton.cdiv(N, 128),)
        _prefill_bm32_bn128[grid](
            a,
            weight,
            out,
            a_scale,
            weight_scale,
            M,
            N,
            K,
            a.stride(0),
            weight.stride(0),
            a_scale.stride(0),
            a_scale.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            GROUP_M=32,
            num_warps=4,
            num_stages=2,
        )
        return out

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
