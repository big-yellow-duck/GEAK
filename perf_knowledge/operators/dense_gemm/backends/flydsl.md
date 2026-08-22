---
title: dense_gemm on FlyDSL — SOTA card
kind: sota_card
operator: dense_gemm
backend: flydsl
gens: [gfx942, gfx950, gfx1200, gfx1201]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp4_e2m1, mxfp4]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/FlyDSL
  - https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
  - https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_authoring_guide.md
  - ROCm/aiter@a6bb4993:aiter/ops/flydsl/gemm_kernels.py
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
---

# dense_gemm × FlyDSL

> RDNA4 uses standalone FlyDSL main's gfx120x wave32/WMMA path. Do not use the AITER integration
> described in older sections of this card: GEAK disables AITER on RDNA4. Start from upstream
> `kernels/gemm/rdna_f16_gemm.py`, use direct `flydsl` imports, and verify `v_wmma*` in generated ISA.

## TL;DR
FlyDSL is AMD's independent **Python/MLIR kernel DSL with instruction-level control** — the productivity
middle ground between Triton and raw assembly. On RDNA4, use upstream FlyDSL directly; its main branch
contains native gfx120x wave32/WMMA kernels and does not require AITER. On CDNA, AITER may optionally host
FlyDSL kernels and dispatch them through its tuned database. Keep those two integration modes separate.

## SOTA implementation
FlyDSL is reached only through the aiter dispatcher, and only when installed. From
`/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
if config["libtype"] == "flydsl":
    if is_flydsl_available():
        flydsl_config = aiter.ops.flydsl.gemm_kernels.get_flydsl_splitk_hgemm_kernel_params(
            config["kernelName"])
        if flydsl_config is None:
            config = None          # named kernel not found -> fall through
    else:
        config = None              # FlyDSL not installed -> next granularity / default
```

The executor (`flydsl_gemm`) asserts **no scaling** for the hgemm path and fuses bias only when dtypes match
(`bias.dtype == inp.dtype and otype in {None, inp.dtype}`), else it adds bias in a follow-up cast.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL split-K hgemm | `aiter/ops/flydsl/gemm_kernels.py::flydsl_hgemm` (+ `get_flydsl_splitk_hgemm_kernel_params`) | gfx942/950; bf16, fp8, A4W4/mxfp4 (MoE) | Kimi-K2.5 fused-MoE (FlyDSL): up to **+162% throughput, −69% TPOT, −65% TTFT** with SGLang+AITER, vendor-reported 2025 | mixed-precision / MoE GEMM; new shapes needing fast iteration to near-asm perf |

## Config space / knobs
From the on-box `flydsl_hgemm` signature (`aiter/ops/flydsl/gemm_kernels.py`). A tuned CSV row carries
`kernelName`; `get_flydsl_splitk_hgemm_kernel_params` decodes it into these:

| param | range / typical | effect | default |
|---|---|---|---|
| `tile_m` / `tile_n` / `tile_k` | 64–256 / 64–256 / 32–128 | per-workgroup output + K tile | 128 / 128 / 64 |
| `split_k` | 1–16 | K-dim split across CUs (skinny/deep-K) | 1 |
| `block_m_warps` / `block_n_warps` | 1–4 / 1–8 | warp grid inside a block | 1 / 4 |
| `n_tile_repeat` | 1–4 | N tiles per workgroup iteration | 1 |
| `persistent_n_tiles` | 1–N | persistent-kernel N tiling | 1 |
| `waves_per_eu` | 0–4 | occupancy hint (0 = compiler-chosen) | 0 |
| `b_to_lds` / `b_to_lds_unroll` | bool / 0–8 | stage B through LDS + unroll | False / 0 |
| `b_preshuffle` | bool | consume pre-shuffled weights (set by `B.is_shuffled`) | True |
| `c_to_lds` | bool | stage C through LDS before store | False |
| `stages` | 1–4 | software-pipeline depth | FIXED_STAGE (2) |
| `async_copy` | bool | use async global→LDS copies | False |
| `kernel_family` | HGEMM / SMALL_M | choose the small-M kernel for decode | HGEMM |

## Numerics / parity
hgemm with **fp32 accumulate**; bias fused when dtype matches (else cast-then-add). The hgemm path
**rejects scale tensors** (`assert scale_a is None and ...`) — scaled/A4W4 GEMM goes through the MoE/quant
FlyDSL paths, gated on task accuracy ([../numerics.md](../numerics.md)). bf16 hgemm is parity-safe vs library.

**fp8 a8w8 block-scale (per-`[128,128]`, arbitrary fp32) parity trap:** do **not** route it through the
native block-scaled MFMA (`mfma_scale_*_f8f6f4`) — its scale is **E8M0 (power-of-two only)** and silently
rounds CK's arbitrary fp32 block scale → parity fail (representational, not tunable;
[../numerics.md](../numerics.md)). Use a **software fp32 post-MFMA scale** core: pin the HW E8M0 scale to
`1.0`, promote+scale after the MFMA. The gfx950 CK→FlyDSL **per-shape** recipe (which software-scale core
to pick + XCD / scheduling levers) is the gated expert skill `flydsl_fp8_blockscale_gemm`.

## Integration (rebind seam)

**RDNA4 (`gfx1200`/`gfx1201`)**: import `flydsl` directly and start from upstream
`kernels/gemm/rdna_f16_gemm.py`. Let runtime detection choose the target or pin
`FLYDSL_GPU_ARCH=gfx1201`; compile/JIT in a subprocess, validate against the frozen oracle, inspect the ISA
for `v_wmma*`, and bind the validated callable directly at the extracted kernel seam. Do not import AITER
or deploy an AITER CSV.

**CDNA (`gfx942`/`gfx950`)**: an optional integration is reached through `aiter.tuned_gemm`: a CSV row
with `libtype=flydsl` plus a `kernelName` that `get_flydsl_splitk_hgemm_kernel_params` resolves and
`is_flydsl_available()` true. This is CDNA integration guidance, not a requirement of FlyDSL itself.

## Pitfalls & anti-patterns
- **Do not test standalone FlyDSL by importing `aiter.ops.flydsl`.** That only tests AITER's wrapper. On
  RDNA4 probe `import flydsl` and compile a native gfx120x example in an isolated subprocess.
- **CDNA AITER integration only:** if FlyDSL is not installed, AITER silently uses CK; verify
  `is_flydsl_available()` before trusting a CDNA `libtype=flydsl` CSV row.
- A flydsl row whose `kernelName` isn't decodable returns `None` → the dispatcher falls through to the next
  `padded_M` granularity / default, so a typo in the CSV silently disables the row.
- Instruction-level control means more knobs than Triton. On RDNA4 start with upstream's native tile and
  tune a bounded grid against the frozen shape suite; never invoke AITER/gradlib. On CDNA, the AITER DB is
  an optional search/dispatch integration.
- hgemm path can't take scales — passing `scale_a/scale_b` raises an assert; use the quant FlyDSL/MoE path.
- **fp8 block-scale ≠ native scaled-MFMA.** Porting CK `gemm_a8w8_blockscale` onto the E8M0 block-scaled
  MFMA fails parity (power-of-two rounding of an arbitrary fp32 scale). Pick a software-fp32-post-MFMA core;
  detail [../numerics.md](../numerics.md), recipe = gated expert skill `flydsl_fp8_blockscale_gemm`.

## How to verify (worked example)

RDNA4 direct path:

```bash
FLYDSL_GPU_ARCH=gfx1201 python -c "import flydsl; print(flydsl.__file__)"
# Then run the upstream rdna_f16_gemm correctness/benchmark entry point in a subprocess,
# compare every frozen case against the oracle, and inspect emitted ISA for v_wmma*.
# Import success alone is not enough: reject a wheel whose authoring API cannot compile that main kernel.
```

CDNA AITER-hosted path only:

```bash
python -c "from aiter.ops.flydsl.utils import is_flydsl_available; print(is_flydsl_available())"
# isolated bench of one tuned kernelName vs hipBLASLt on the same (M,N,K)
python gradlib/gradlib/gemm_tuner.py --indtype bf16 --libtype flydsl -i shapes.csv -o /tmp/fly.csv
# e2e: deploy via aiter env + confirm libtype is flydsl in the log
AITER_CONFIG_GEMM_BF16=/tmp/fly.csv AITER_LOG_TUNED_CONFIG=1 <launch> ; grep 'libtype is flydsl' server.log
```

## Alternatives / cross-links
[[operators/dense_gemm/backends/aiter]] (dispatch + deploy) · [[operators/dense_gemm/backends/triton]]
(easier to author, lower ceiling) · [[operators/dense_gemm/backends/hipblaslt]] (library default) ·
[[operators/dense_gemm/backends/ck]] (the fallback) · [[operators/grouped_gemm_moe/backends/aiter]]
(FlyDSL MoE) · language deep-dive `languages/flydsl/` (P1) · authoring how-to: [[languages/flydsl/authoring_gemm_levers]] (tiling / LDS / MFMA-loop / epilogue when writing a GEMM `@flyc.kernel`) + [[languages/flydsl/authoring_tile_programming]] (CuTe tile model) + [[languages/flydsl/authoring_optimization]] (structure-first workflow) + [[languages/flydsl/debugging]] (correctness / NaN / hang triage).

## Sources
- On-box: `/sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py` (`flydsl_hgemm` signature),
  `aiter/tuned_gemm.py` (flydsl branch) — `ROCm/aiter@a6bb4993`.
- Kimi-K2.5 FlyDSL fused-MoE numbers (+162% tput / −69% TPOT / −65% TTFT): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
