---
title: dense_gemm on direct HIP — RDNA4 WMMA card
kind: sota_card
operator: dense_gemm
backend: hip
gens: [gfx1200, gfx1201]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-08-22
sources:
  - https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
  - https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
  - https://rocm.docs.amd.com/projects/rocWMMA/en/develop/supported-hardware.html
---

# dense_gemm × direct HIP on RDNA4

## TL;DR

Use a direct gfx12 HIP/rocWMMA kernel when fusion, unusual layouts, or exact
fragment scheduling justify owning the kernel. gfx120x is wave32 and uses WMMA,
not CDNA MFMA. A plain dense GEMM should still be compared with the installed
rocBLAS/hipBLASLt path, but GEAK does not route it through AITER on RDNA4.

## Architecture contract

- Compile natively: `hipcc --offload-arch=gfx1201 -O3` (or gfx1200).
- Query `warpSize`; the native path is 32 lanes.
- One CU is half a WGP and contains two SIMD32s; a workgroup resides on one WGP.
- The official table reports 128 KiB LDS across the CU pair, while the live HSA
  GROUP pool and FlyDSL expose a 64 KiB per-workgroup cap on gfx1201. Query the
  runtime/compiler and use 64 KiB for portable tile planning unless proven otherwise.
- Record both rocminfo physical CUs and WGP scheduler units. On an R9700 those
  are 64 and 32 respectively, while PyTorch calls the latter
  `multi_processor_count`. Start persistent-grid sweeps at one and two
  workgroups per WGP; the architecture id alone is insufficient.
- Build bandwidth roofs from same-box GDDR measurements; there is no MI HBM/XCD.

## Matrix path

The base floating-point operation is wave32 16x16x16 FP16/BF16 with FP32
accumulation. Raw HIP can use gfx12 builtins such as:

```cpp
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(...)
```

Prefer rocWMMA fragments when their abstraction fits. RDNA4 uses a new v8
operand VGPR layout; never reuse gfx11 hand-packed fragments. Inspect the code
object and require `v_wmma*` in the hot loop before crediting matrix-core FLOPS.

## Tuning dimensions

- workgroup: 64 / 128 / 256 threads (2 / 4 / 8 wave32 waves);
- output WMMA fragments per wave;
- K-stage depth and single/double LDS buffers;
- cooperative global→LDS vector width and LDS padding;
- one vs two resident workgroups per WGP, measured from compiled VGPR/LDS use;
- split-K only when the output grid underfills detected CUs;
- fused bias/activation/quant epilogue to avoid a second GDDR pass.

## Correctness and performance gates

1. Run the immutable oracle on every case/dtype.
2. Dump compiler resource remarks; reject spills and LDS over-allocation.
3. Inspect ISA for gfx12 `v_wmma*` and the expected input datatype.
4. Compare per-case latency with the frozen input kernel and library baseline.
5. Use a scalar/vector deletion control if the matrix-core path appears not to
   move timing—it may have fallen back or be launch/memory bound.

## RDNA4 prohibitions

- no `v_mfma_*`, `matrix_instr_nonkdim`, wave64 reduction, HBM, or XCD advice;
- no AITER import/tuner/config deployment;
- no assumption that an installed CK/CK-Tile instance has an RDNA code object.
