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
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsm: tl.constexpr,
    stride_bsk: tl.constexpr,
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

    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * stride_bm + offs_k[:, None]
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + pid_n * stride_bsm

    accumulator = tl.zeros((32, 128), dtype=tl.float32)
    for k_block in range(tl.cdiv(K, 128)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + k_block * stride_ask, mask=offs_m < M, other=0.0)
        b_s = tl.load(bs_ptrs + k_block * stride_bsk)
        combined_scale = a_s * b_s
        accumulator += tl.dot(a, b) * combined_scale[:, None]
        a_ptrs += 128
        b_ptrs += 128

    out_m = pid_m * 32 + tl.arange(0, 32)
    out_n = pid_n * 128 + tl.arange(0, 128)
    tl.store(
        C + out_m[:, None] * N + out_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=out_m[:, None] < M,
    )


@triton.jit
def _prefill_bm64_bn128_fused_scale(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsm: tl.constexpr,
    stride_bsk: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """One 64x128 output tile with factored row/tile scales."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 64)
    num_pid_n = tl.cdiv(N, 128)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * stride_bm + offs_k[:, None]
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + pid_n * stride_bsm

    accumulator = tl.zeros((64, 128), dtype=tl.float32)
    for k_block in range(tl.cdiv(K, 128)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + k_block * stride_ask, mask=offs_m < M, other=0.0)
        b_s = tl.load(bs_ptrs + k_block * stride_bsk)
        combined_scale = a_s * b_s
        accumulator += tl.dot(a, b) * combined_scale[:, None]
        a_ptrs += 128
        b_ptrs += 128

    out_m = pid_m * 64 + tl.arange(0, 64)
    out_n = pid_n * 128 + tl.arange(0, 128)
    tl.store(
        C + out_m[:, None] * N + out_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=out_m[:, None] < M,
    )


@triton.jit
def _prefill_bm80_bn128_shared_b(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsm: tl.constexpr,
    stride_bsk: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """A 64-row and a 16-row accumulator band share each B tile."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 80)
    num_pid_n = tl.cdiv(N, 128)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m0 = pid_m * 80 + tl.arange(0, 64)
    m1 = pid_m * 80 + 64 + tl.arange(0, 16)
    n = pid_n * 128 + tl.arange(0, 128)
    k = tl.arange(0, 128)
    a0_ptrs = A + m0[:, None] * stride_am + k[None, :]
    a1_ptrs = A + m1[:, None] * stride_am + k[None, :]
    b_ptrs = B + n[None, :] * stride_bm + k[:, None]
    as0_ptrs = As + m0 * stride_asm
    as1_ptrs = As + m1 * stride_asm
    bs_ptrs = Bs + pid_n * stride_bsm

    acc0 = tl.zeros((64, 128), dtype=tl.float32)
    acc1 = tl.zeros((16, 128), dtype=tl.float32)
    for k_block in range(tl.cdiv(K, 128)):
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs + k_block * stride_bsk)
        a0 = tl.load(a0_ptrs, mask=m0[:, None] < M, other=0.0)
        a1 = tl.load(a1_ptrs, mask=m1[:, None] < M, other=0.0)
        s0 = tl.load(as0_ptrs + k_block * stride_ask, mask=m0 < M, other=0.0)
        s1 = tl.load(as1_ptrs + k_block * stride_ask, mask=m1 < M, other=0.0)
        acc0 += tl.dot(a0, b) * (s0 * b_s)[:, None]
        acc1 += tl.dot(a1, b) * (s1 * b_s)[:, None]
        a0_ptrs += 128
        a1_ptrs += 128
        b_ptrs += 128

    tl.store(
        C + m0[:, None] * N + n[None, :],
        acc0.to(tl.bfloat16),
        mask=m0[:, None] < M,
    )
    tl.store(
        C + m1[:, None] * N + n[None, :],
        acc1.to(tl.bfloat16),
        mask=m1[:, None] < M,
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

    use_bm32 = M == 523 and N == 8192
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

    use_bm64 = (
        (M in (72, 138, 139) and not (N == 5120 and K == 8704))
        or (M == 249 and (K == 3072 or N in (7168, 17408)))
        or (M == 277 and N == 5120)
        or (M == 523 and N in (5120, 7168, 17408))
        or (M == 784)
    )
    if use_bm64:
        out = torch.empty((M, N), device=a.device, dtype=output_dtype)
        use_bm80 = M == 784
        block_m = 80 if use_bm80 else 64
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, 128),)
        prefill_kernel = (
            _prefill_bm80_bn128_shared_b
            if use_bm80
            else _prefill_bm64_bn128_fused_scale
        )
        prefill_kernel[grid](
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
