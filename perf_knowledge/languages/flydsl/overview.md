---
title: FlyDSL — AMD's independent Python/MLIR kernel DSL — overview
kind: language
gens: [gfx942, gfx950, gfx1200, gfx1201]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1, mxfp4]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/ROCm/FlyDSL
  - https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
  - https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_authoring_guide.md
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - https://github.com/ROCm/aiter
  - /sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py
  - /sgl-workspace/aiter/aiter/ops/flydsl/kernels/splitk_hgemm.py
---

# FlyDSL — overview

## TL;DR
**FlyDSL** is AMD's independent **Python/MLIR kernel DSL with instruction-level control**. AITER is one
consumer/integration of FlyDSL, not its runtime owner. Unlike Triton (which hides the matrix core
and lets a compiler pick the schedule), FlyDSL lets you emit target-specific **ROCDL intrinsics** —
MFMA on CDNA or WMMA on RDNA, data movement, `sched_mfma`/
`sched_vmem`/`sched_group_barrier` (hand-built software pipeline), `ds_bpermute`/`ds_swizzle`,
`s_setprio`, `s_waitcnt` — while still writing Python. Its IR is **FLIR (Flexible Layout IR)**, a
CuTe-style **(Shape, Stride)** layout algebra for tiling/swizzling/vectorization. Think "Triton's
ergonomics with CK's control."

FlyDSL is the engine behind AMD's published **+162% throughput / −69% TPOT / −65% TTFT** Kimi-K2.5
result on MI300X. Separately, upstream FlyDSL main has direct architecture detection and per-subtarget
lowering for CDNA3, CDNA4, gfx11, gfx120x, and gfx1250.

### RDNA4 status

Upstream main lists Radeon AI PRO R9700 (`gfx1201`) as a verified platform and provides a native
`gfx120*` **wave32/WMMA** path plus RDNA GEMM tests/examples. Use `get_rocm_arch()` / `is_rdna_arch()`
and `get_warp_size()` rather than hard-coding CDNA behavior. gfx120x uses the new v8-operand WMMA ABI;
it is not compatible with gfx11 fragment packing and is not gfx1250.

In GEAK's RDNA4 path, use the standalone `flydsl` package and upstream direct kernels. **Do not import
`aiter.ops.flydsl`**: AITER is disabled on RDNA4 due to its still-experimental, crash-prone non-FlyDSL
components. This does not disable FlyDSL. An import/architecture probe alone is not an availability
test: released wheels may recognize gfx1201 while lagging the Python surface used by main's RDNA GEMM.
Compile and parity-run that native kernel in a subprocess before admitting the backend.

## Where it fits
| Use FlyDSL when | Reach elsewhere when |
|---|---|
| You want **instruction-level control** (sched, direct-to-LDS, preshuffle) but in Python | Plain prototype → `triton` |
| Authoring **fused MoE** / split-K HGEMM / W4A16 mixed GEMM for aiter | Generic templated GEMM/attn → `ck_tile` |
| A tuned shape already has a `flydsl` entry in the aiter GEMM table | One-off shape with no tuned config → library default |
| You need the +X% the compiler leaves on the table, without dropping to raw HIP/asm | Last % beyond FlyDSL → `asm`/HipKittens |

## On-box: where it lives and what it exposes
The installed version/path are environment-dependent; record `flydsl.__file__`, package metadata, and
the exact upstream revision used for any vendored main-branch kernel:
```
flydsl/
├── expr/          # the DSL surface: arith, gpu, vector, math, buffer_ops, rocdl/  ← intrinsics
├── compiler/      # ast_rewriter, jit_function/executor, kernel_function, backends, llvm_options
├── _mlir/         # MLIR Python bindings (ir, dialects: fly, llvm, memref, scf, gpu, rocdl)
├── runtime/       # device.py (get_rocm_arch -> 'gfx942'/'gfx950'), device_runtime
├── utils/         # smem_allocator (SmemAllocator, SmemPtr), ...
└── autotune.py    # Config(num_warps, waves_per_eu, maxnreg, **kwargs) + @autotune
```
aiter's FlyDSL kernels live at `ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/`:
`gemm_kernels.py` (HGEMM/preshuffle APIs), `moe_kernels.py`, `linear_attention_kernels.py`, and
`kernels/` (the actual DSL bodies: `splitk_hgemm.py`, `small_m_hgemm.py`, `preshuffle_gemm.py`,
`mfma_preshuffle_pipeline.py`, `moe_gemm_2stage.py`, `mixed_moe_gemm_2stage.py`, `silu_and_mul_fq.py`,
`gdr_decode.py`, `reduce.py`).

## The mental model (vs Triton)
- **Triton:** `tl.dot` → compiler picks MFMA shape, LDS swizzle, and schedule. You tune via knobs.
- **FlyDSL:** *you* call `rocdl.mfma_f32_16x16x16bf16_1k(...)`, *you* place
  `rocdl.raw_ptr_buffer_load_lds(...)` and `rocdl.sched_mfma(n)/sched_vmem(n)/sched_group_barrier(...)`,
  *you* allocate LDS via `SmemAllocator` and swizzle with `swizzle_xor16(row, col, k_blocks16)`. FLIR
  layouts handle the index math; ROCDL ops handle the hardware. The compiler still does register
  allocation, canonicalization/CSE, and GPU-to-ROCDL lowering.

## What it targets
`get_rocm_arch()` auto-detects the target; `FLYDSL_GPU_ARCH` overrides it. Upstream targets include
gfx942, gfx950, and gfx1201. The older AITER wrapper notes below are CDNA integration facts only:
- `KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"` → **async-copy/direct-to-LDS is gfx950-default,
  off on gfx942** (matches Triton/HIP: `global_load_lds` is gfx942 but the async-LDS path is gfx950).
- LDS budget from `addressable_lds_bytes_for_gfx`: **65536 B (gfx942)**, **163840 B (gfx950)**.
- ROCDL exposes both FNUZ and OCP MFMA + block-scaled `mfma_scale_f32_16x16x128_f8f6f4` (CDNA4 MXFP).

For gfx120x, use upstream's GFX120X atoms and `kernels/gemm/rdna_f16_gemm.py`: wave32, WMMA, a
64 KiB per-CU LDS model, and gfx12-specific fragment ABI. Do not apply the three CDNA wrapper bullets above.

## Deep-dive map

**Using the library** (aiter's built-in flydsl GEMM primitives):
- [deep.md](deep.md) — FLIR layout algebra, the ROCDL intrinsic surface, compile/JIT flow, LDS/sched.
- [patterns.md](patterns.md) — `flydsl_hgemm` usage, preshuffle, split-K, small-M, the sched pipeline.
- [knobs.md](knobs.md) — the full `flydsl_hgemm` knob set (verified against source) + autotune.
- [kernel_families.md](kernel_families.md) — HGEMM / small-M / preshuffle / 2-stage MoE / GDR decode.

**Authoring your own `@flyc.kernel`** (ingested from the FlyDSL authoring skill — reference how-to):
- [authoring_tile_programming.md](authoring_tile_programming.md) — write a first correct kernel (CuTe-style tile model, the 4 patterns, MFMA reference).
- [authoring_optimization.md](authoring_optimization.md) — structure-first optimization workflow (fusion → LDS → MFMA-loop → tuning).
- [authoring_gemm_levers.md](authoring_gemm_levers.md) — GEMM-specific levers (tiling / LDS staging / swizzle / epilogue).
- [debugging.md](debugging.md) — correctness/stability/hang triage (NaN / zeros / mismatch / compile / hang).

## Sources
- Kimi-K2.5 optimization with FlyDSL (FLIR, instruction-level control, +162% throughput): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- aiter (engine that hosts FlyDSL): https://github.com/ROCm/aiter
- Standalone FlyDSL main (gfx1201 verified): https://github.com/ROCm/FlyDSL
- Architecture guide (gfx120x wave32/WMMA): https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
- FlyDSL HGEMM API & arch gating: ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/gemm_kernels.py
- ROCDL intrinsic surface (mfma/sched/buffer_load_lds): flydsl 0.1.5 @ /opt/venv/lib/python3.10/site-packages/flydsl/expr/rocdl/
- DSL body (mfma/swizzle/sched primitives in a real kernel): ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/kernels/splitk_hgemm.py
