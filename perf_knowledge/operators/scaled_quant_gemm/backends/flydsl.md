---
title: scaled_quant_gemm on FlyDSL — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: flydsl
gens: [gfx942, gfx950, gfx1200, gfx1201]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-08-28
sources:
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/kernels/preshuffle_gemm.py
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/test_flydsl_moe_a4w4.py
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/gemm_kernels.py
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - big-yellow-duck/FlyDSL@eed78c6d:lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp
  - https://github.com/ROCm/FlyDSL/blob/main/kernels/gemm/rdna_f16_gemm.py
---

# scaled_quant_gemm × FlyDSL

> **Architecture split:** the established preshuffle/MXFP material below is CDNA/AITER-specific. On
> gfx1200/gfx1201 use standalone FlyDSL, OCP FP8, wave32 WMMA, and the raw-weight/software-scale path
> documented in [`../../../languages/flydsl/rdna4.md`](../../../languages/flydsl/rdna4.md). GEAK does
> not permit AITER on RDNA4.

## RDNA4 capability and maturity

The gfx120x compiler path can lower `16x16x16` FP8 E4M3FN WMMA with FP32 accumulation. An on-box
R9700 prototype also validated a vLLM-compatible raw-weight block-scale contract for M1--M64:
contiguous A, arbitrary padded B row stride, FP32 A/B scales per K128, FP32 accumulation, and one BF16
conversion. It passed random parity, route boundaries, stream, fresh-output, and graph-replay tests.

That does **not** make the CDNA preshuffle kernel an RDNA route, and it does not establish a broad-M
winner. The broad-prefill GEAK campaign promoted no FlyDSL patch and measured 0.9895x weighted against
Triton under a timing run that was itself flagged for a missing receipt. The actionable conclusion is:

- decode/small-M has a real register-fed FP8 WMMA starting implementation;
- broad prefill needs an LDS-tiled, cross-wave operand-reuse pipeline before scheduler micro-tuning;
- performance claims remain per-shape and require a new paired receipt.

## TL;DR
On CDNA, the **scaled** (dequant-fused) GEMM path is where FlyDSL is SOTA: a separate `preshuffle_gemm` kernel family
consumes quantized A/W (**fp8 / int8 / int4 / fp4**) plus dequant scales and produces bf16/fp16. The dense
hgemm path explicitly **rejects scales** — scaling lives here. On **CDNA4 (gfx950)** the fp4 path uses the
block-scaled `mfma_scale_f32_16x16x128_f8f6f4` MXFP MFMA; on gfx942 it runs the fp8/int8 preshuffle kernel.
This is the family Kimi-K2.5 used for MoE GEMM (vendor: up to +162% throughput).

## CDNA SOTA implementation
A8W8 (fp8/int8) and W4 (MXFP4) share one compiler, `compile_preshuffle_gemm_a8`; `compile_preshuffle_gemm_w4`
just delegates with `in_dtype="fp4"` and is **gfx950-only**. From
`/sgl-workspace/aiter/aiter/ops/flydsl/kernels/preshuffle_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
def compile_preshuffle_gemm_w4(*, N, K, tile_m, tile_n, tile_k,
                               a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
                               lds_stage=2, ...):
    """MXFP4 preshuffle GEMM — delegates to compile_preshuffle_gemm_a8 with fp4 config."""
    if a_dtype == "fp8":
        raise NotImplementedError("fp8-A not yet supported with MXFP4 kernel (op_sel_a overflow)")
    if str(get_hip_arch()) != "gfx950":
        raise RuntimeError(f"FP4 GEMM requires gfx950, got {get_hip_arch()}")
    inner = compile_preshuffle_gemm_a8(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                                       in_dtype="fp4", lds_stage=lds_stage, out_dtype=out_dtype, ...)
    return inner
```

Inside `compile_preshuffle_gemm_a8` the block-scaled MFMA is selected on CDNA4 for sub-byte types
(`use_mfma_scale_128 = gpu_arch.startswith("gfx95") and not int8/int4/f16/bf16`), requires `tile_k % 128 == 0`,
and emits `rocdl.mfma_scale_f32_16x16x128_f8f6f4` with `cbsz/blgp = 4` and `pack_M = 2` for fp4. int8 uses
`rocdl.mfma_i32_16x16x32i8`. The driver `flydsl_preshuffle_gemm_a8` (in `gemm_kernels.py`) takes
`XQ, WQ, x_scale, w_scale, Out` and dispatches in_dtype by `XQ.dtype` (fp8 / int8).

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| gfx120x raw-weight software-scale WMMA | on-box RDNA4 prototype; atom in `big-yellow-duck/FlyDSL@eed78c6d` | gfx1200/1201; OCP fp8_e4m3fn + arbitrary FP32 K128 scales | M1--M64 parity and HIP comparisons; broad-M Triton win not established | decode/small-M seed; broad-M requires a new LDS reuse implementation |
| A8W8 preshuffle GEMM | `preshuffle_gemm.py::compile_preshuffle_gemm_a8` | gfx942/950; fp8/int8 → bf16/fp16 | no isolated flydsl number; folded into aiter a8w8 bpreshuffle GEMM tune | per-tensor/row fp8/int8 GEMM with preshuffled W |
| W4 / MXFP4 block-scaled GEMM | `preshuffle_gemm.py::compile_preshuffle_gemm_w4` (→ a8 with fp4) | **gfx950 only**; fp4 (per_1x32) → bf16/fp16 | Kimi-K2.5 fused-MoE (FlyDSL, vendor): up to **+162% throughput, −69% TPOT, −65% TTFT** (SGLang+AITER, 2025) | MXFP4 MoE / dense low-bit GEMM |

## Config space / knobs
From `compile_preshuffle_gemm_a8` signature + the tune catalog
`gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`:

| param | range / source | effect | default |
|---|---|---|---|
| `tile_m`/`tile_n`/`tile_k` | base-tile catalog (`_base_tiles_lds2_common`, …); fp4 needs `tile_k=128` or `≥128` | output + K tile | per tune row |
| `in_dtype` | `fp8 \| int8 \| int4 \| fp16 \| bf16 \| fp4` | input quant type | "fp8" |
| `out_dtype` | `fp16 \| bf16` | output type | "fp16" |
| `lds_stage` | `1 \| 2` (fp4 tile_k=128 → must be 2) | ping/pong LDS depth | 2 |
| `use_cshuffle_epilog` | `0 \| 1` | LDS C-shuffle epilogue (see [[operators/gemm_epilogue_fused/backends/flydsl]]) | 0 |
| `use_async_copy` | `0 \| 1` (gfx942 async load = 4B, else 16B) | async A g→LDS | 0 |
| `waves_per_eu` | `0=none, 1..4`; tune sweep `(0,1,2,3,4)` | occupancy hint | None |
| `dsrd_preload`/`dvmem_preload` | `-1` = auto from `_TILE_PRELOAD_TABLE` (gfx950 fp8/int8) | LDS-read / global-load preload | -1 |

The tune module carries per-arch base-tile lists (`kernels_list_942`, `kernels_list_950`), prunes by
estimated LDS (`preshuffle_gemm_estimated_lds_bytes` vs `max_lds_bytes_for_tune()`) and a VGPR-pressure
`waves_per_eu` cap (`_estimate_max_wpe`). Default kernels: gfx942 `(128,128,128,lds2,wpe2)`, gfx950
`(128,256,256,lds2,wpe2)`.

## Numerics / parity
On gfx120x, arbitrary FP32 K128 scales are applied in software after completing the eight K16 FP8
WMMAs in each K128 block. Sum scaled blocks in FP32 and cast once. Do not substitute E8M0 scaled-WMMA,
scale each K16 partial independently, or scale after summing all K blocks.

fp32 accumulate. Scales are applied via the MXFP block-scaled MFMA on gfx950 (block size 32 — the scale
layout is `(c_mn1, c_k1, 4, 16)` with `scale_block_size=32`, see [[operators/layout_shuffle/backends/flydsl]]).
`tile_k_bytes` must be divisible by 64; fp4 requires `tile_k=128` (or `k_unroll≥1` for
`mfma_scale_f32_16x16x128`). a4w4 MoE test tolerances: `atol=1.0, rtol=0.05, pass≥95%` per stage, `≥90%`
e2e (`test_flydsl_moe_a4w4.py`, `QuantType.per_1x32`, `fp4x2`).

## Integration (rebind seam)
**RDNA4**: use direct `flydsl` imports, current PyTorch stream, cached JIT modules, and one operator
dispatch. Raw B may have padded row stride but K must be contiguous. Do not preshuffle or persist model
weights unless the operator contract explicitly owns that transformation. Compile the exact gfx120x
source in a subprocess, then prove the FP8 WMMA in ISA.

**CDNA**: the two AITER seams are described below. They are not RDNA4 options.

Two seams: (1) the a8w8 bpreshuffle GEMM tune (`gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py` →
`kernelInstance.name` = `flydsl_bpreshuflle_<m>x<n>x<k>_<qa>_<qw>_<dt>_<lds>x<csh>x<async>x<wpe>_default`)
selecting a `solidx`; (2) `flydsl_preshuffle_gemm_a8(XQ,WQ,x_scale,w_scale,Out,...)` exported from
`aiter.ops.flydsl` and lazily compiled via `_get_compile_fn()` (logs `"[FlyDSL] loaded preshuffle GEMM
compiler"`; on absence falls back to CK/CKTile). `is_flydsl_available()` gates all of it.

## Pitfalls & anti-patterns
- **Do not send gfx120x through the AITER preshuffle path.** Its formats, integration seam, architecture
  tables, and MFMA schedule are CDNA-specific.
- **A working FP8 atom is not a broad-M kernel.** At prefill M, add LDS reuse and a measured pipeline;
  the register-fed small-M route leaves the main Triton advantage intact.
- gfx120x uses OCP `float8_e4m3fn`, wave32 WMMA, and a 64 KiB portable LDS cap. FNUZ/MFMA/gfx950
  assumptions are correctness or launch failures, not tuning choices.
- **fp4/MXFP4 is gfx950-only** — `compile_preshuffle_gemm_w4` raises on gfx942; fp8-A + MXFP4 is
  `NotImplementedError` (op_sel_a overflow).
- `tile_k_bytes % 64 != 0` raises; fp4 with `tile_k != 128` and `k_unroll < 2` raises.
- Don't route scaled GEMM through `flydsl_hgemm` — it asserts no scale; use the preshuffle family.
- LDS over the arch limit silently drops a tune candidate (the tune prunes by `preshuffle_gemm_estimated_lds_bytes`).

## How to verify
```bash
# gfx120x direct path
FLYDSL_GPU_ARCH=gfx1201 pytest -q tests/kernels/test_rdna4_wmma_atom.py
# Then run the operator parity/graph suite and inspect ISA for v_wmma.*fp8.

# CDNA AITER path
python -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"     # fp4 path needs gfx950
pytest -q aiter/ops/flydsl/test_flydsl_moe_a4w4.py                              # a4w4 stage1/stage2/e2e
```

## Alternatives / cross-links
[[operators/scaled_quant_gemm/backends/aiter]] (sota dispatch) · [[operators/scaled_quant_gemm/backends/triton]] ·
[[operators/layout_shuffle/backends/flydsl]] (preshuffle B + scale layouts) ·
[[operators/gemm_epilogue_fused/backends/flydsl]] (C-shuffle epilogue) ·
[[operators/grouped_gemm_moe/backends/aiter]] (FlyDSL MoE GEMM).

**Authoring / optimizing a FlyDSL GEMM `@flyc.kernel`** (write from scratch OR port ck→flydsl):
[[languages/flydsl/authoring_gemm_levers]] (tiling / LDS / XCD-swizzle / epilogue) ·
[[languages/flydsl/authoring_optimization]] (structure-first workflow) ·
[[languages/flydsl/authoring_tile_programming]] (CuTe tile model) ·
[[languages/flydsl/debugging]] (correctness / NaN / hang triage).
RDNA4 authors must additionally read [[languages/flydsl/rdna4]].

## Sources
- On-box: `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/preshuffle_gemm.py`
  (`compile_preshuffle_gemm_a8`, `compile_preshuffle_gemm_w4`, `use_mfma_scale_128` /
  `mfma_scale_f32_16x16x128_f8f6f4`), `gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py` (tile catalog,
  `kernelInstance`, LDS/wpe pruning), `gemm_kernels.py` (`flydsl_preshuffle_gemm_a8` driver),
  `test_flydsl_moe_a4w4.py` (tolerances) — `ROCm/aiter@a6bb4993`, flydsl 0.1.5.
- Kimi-K2.5 FlyDSL fused-MoE numbers (vendor): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- RDNA4 capability, source hashes, and positive/negative receipts:
  [`../../../languages/flydsl/rdna4.md`](../../../languages/flydsl/rdna4.md)
