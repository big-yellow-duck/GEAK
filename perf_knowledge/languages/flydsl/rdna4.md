---
title: FlyDSL on RDNA4 — gfx120x wave32/WMMA authoring and capability card
kind: language
gens: [gfx1200, gfx1201]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [prefill, decode, both]
status: experimental
updated: 2026-09-07
sources:
  - https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
  - https://github.com/ROCm/FlyDSL/blob/main/kernels/gemm/rdna_f16_gemm.py
  - https://github.com/ROCm/FlyDSL/commit/3c03e97919bedbeb95ea803baed089c3725eabb6
  - https://github.com/ROCm/FlyDSL/blob/main/kernels/gemm/rdna_fp8_preshuffle_gemm.py
  - https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
  - https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
---

# FlyDSL on RDNA4 (`gfx1200` / `gfx1201`)

This is the RDNA4 companion to the CDNA-oriented FlyDSL authoring material. Read it whenever the
detected target is `gfx1200` or `gfx1201`; do not translate the MFMA/wave64 recipes mechanically.

The short version: the framework and lowering path are real, but capability, correctness coverage,
and performance maturity are separate facts. Upstream has native wave32/WMMA dense-GEMM and FP8
preshuffle paths, and commit `3c03e97919bedbeb95ea803baed089c3725eabb6` merged the gfx120x FP8/BF8
WMMA atom. GEAK's raw-weight block-scaled FP8 prototype is a different operator contract and remains
an uncommitted R9700 campaign snapshot. Treat the sources and receipts below as a starting stack, not
as a claim that every RDNA4 FlyDSL route is competitive.

## Capability ledger

| Layer | gfx120x state | Evidence | Agent consequence |
|---|---|---|---|
| Architecture selection | available | `get_rocm_arch()`, `is_rdna_arch()`, `get_warp_size()` and `FLYDSL_GPU_ARCH` | detect first; pin `gfx1201` only for a box-specific run |
| FP16/BF16 dense GEMM | upstream reference | `kernels/gemm/rdna_f16_gemm.py` | use this as the broad-M LDS/WMMA seed |
| FP8/BF8 WMMA atom | merged upstream in `ROCm/FlyDSL@3c03e979` | `GFX120X/MmaAtom.cpp`, MLIR positive/negative tests, and `test_rdna4_wmma_atom.py` | pin a revision containing that commit and require the intended `v_wmma` in ISA |
| Upstream FP8 preshuffle GEMM | upstream reference | `kernels/gemm/rdna_fp8_preshuffle_gemm.py` and `test_rdna_gemm.py` | reuse scheduling ideas only when per-token A scales, per-channel B scales, and preshuffled B match the contract |
| Raw-weight FP8 block scale, M1--M64 | parity-tested prototype | on-box source/benchmark snapshot listed below | usable evidence for decode/small-M authoring; not a broad-prefill template |
| Raw-weight FP8 block scale, broad M | correctness route exists, performance open | GEAK broad-prefill campaign finished at 0.9895x weighted vs Triton and promoted no patch | fund LDS reuse/pipeline work before micro-tuning |
| AITER integration | unsupported in GEAK RDNA4 policy | architecture gate and crash history | import standalone `flydsl`; never use `aiter.ops.flydsl` |
| gfx1250 TDM/scaled WMMA | different subtarget | gfx1250-only atoms and LDS model | never enable by matching the `gfx12` prefix |

The broad-prefill figure above is a negative result, not a backend ban. That run was also flagged by
its timing-receipt gate; use it to reject unsupported performance claims and to prioritize missing
structure, not as a precise roofline number.

## Architecture facts that change the kernel

- Native execution is wave32. One 128-thread workgroup contains four waves.
- The gfx120x floating-point WMMA atom is `16x16x16` with FP32 accumulation. Each lane holds eight
  A elements, eight B elements, and eight FP32 accumulator slots for one atom. This is the v8 operand
  ABI; copying gfx11 fragment packing is wrong.
- The tested atom accepts FP16, BF16, or OCP FP8 E4M3FN A/B with FP32 accumulation. It does not accept
  sign/clamp modifiers, and `16x16x32` is not a gfx120x shape.
- Use WGP count for scheduling/fill reasoning and retain physical CU count separately. On the R9700
  used for the receipts, those are 32 WGPs and 64 physical CUs.
- Treat 64 KiB as the portable per-workgroup LDS cap. RDNA4 LDS is 32-bank and must be measured with
  wave32 access patterns.
- This is GDDR6 plus Infinity Cache, not HBM/XCD. CDNA XCD swizzles and HBM roof constants do not
  transfer.

## Two correct starting structures

### Broad-M dense or prefill GEMM: tile through LDS

Upstream `rdna_f16_gemm.py` is the primary skeleton:

- `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=32`;
- four waves in a 2x2 wave layout;
- 128-bit global loads;
- two padded LDS stages, about 40 KiB total;
- register-to-LDS ping-pong with a workgroup barrier;
- layout-derived `make_tiled_copy_A/B/C`, rather than handwritten lane math;
- an L2-local workgroup mapping controlled by `group_m`.

For a broad FP8 kernel, retain this structure but change the K contract deliberately. A raw-weight
block-scaled GEMM with K128 scale groups needs eight K16 WMMAs per scaled partial. Complete those
WMMAs in FP32, apply the independent FP32 A/B scales, add to the FP32 total, and cast once. The scale
boundary is part of correctness and cannot be moved to improve scheduling.

The current small-M FP8 prototype is register-fed and does not provide the cross-wave A/B LDS reuse
needed at M256. For prefill, build the LDS reuse path before adjusting scheduler counts.

### Decode or very small M: register-fed WMMA

For M1--M64, the measured prototype streams 64-bit FP8 fragments into the gfx120x atom and reuses
each B fragment across one or more row fragments. Its useful general patterns are:

- use 64 or 128 threads depending on the number of output-column fragments;
- keep route thresholds coarse and cover adjacent M/N/K cases;
- batch independent VMEM loads before dependent WMMAs when registers permit;
- split K only when the resulting grid fill pays for the LDS reduction;
- use a small LDS-only signal/wait fence for split-K handoff rather than a global-memory fence;
- accept arbitrary padded B row stride while keeping K contiguous.

Do not copy its exact M1/M2 routing table into a broad-M target.

## Synchronization and scheduling

Generic `gpu.barrier()` is correct for a whole-workgroup LDS handoff, but it may carry more ordering
than a specialized path needs. The validated gfx12 helper sequence is:

1. `s_wait_dscnt(0)` for the producing wave's required LDS writes;
2. `s_barrier_signal(-1)`;
3. independent address work where legal;
4. `s_barrier_wait(-1)` before consuming peer LDS data.

Use that split signal/wait only when every wave executes the same barrier instance. Never place it
behind a divergent branch. `sched_vmem`, `sched_mfma`, and `sched_barrier` counts must be derived from
the current unrolled region; the measured M1 route used 40 VMEM loads and 32 WMMAs per K128 group,
but that is not a universal setting.

## FP8 block-scale numerical contract

This GEAK target is not upstream's `rdna_fp8_preshuffle_gemm.py`. That upstream kernel uses a raw A,
preshuffled B, per-token A scale, and per-channel B scale. The target here owns raw B and arbitrary
FP32 A/B scales at K128 boundaries; silently substituting the upstream layout changes the operator.

For A `[M,K]`, raw B `[N,K]`, A scale `[M,K/128]`, and B scale `[N/128,K/128]`:

```text
total = 0 (FP32)
for kb in K/128:
    partial = dot_fp8_fp8_fp32(A[:, kb*128:(kb+1)*128],
                               B[:, kb*128:(kb+1)*128])
    total += partial * a_scale[:, kb] * b_scale[:, kb]
out = bf16(total)
```

Arbitrary FP32 block scales are not E8M0 powers of two. Do not route this contract through a native
scaled-MMA encoding that rounds the scales. Do not scale each K16 WMMA independently, and do not sum
all K before applying scales.

## Search order for a serious RDNA4 campaign

1. Prove the gfx120x atom: compile, run parity, and find `v_wmma*` in ISA.
2. Match the reuse structure to the regime: LDS-tiled for broad M; register-fed for tiny M.
3. Establish one correct stage before adding ping-pong/prefetch.
4. Sweep 64/128/256 threads and WMMA repeat geometry while recording VGPR, SGPR, LDS, and scratch.
5. Measure LDS padding or swizzle with wave32 access; do not import a CDNA bank rule.
6. Tune prefetch distance and scheduler groups only after the instruction census is known.
7. Compare direct and LDS-reordered epilogues; keep the simpler one unless stores are demonstrably
   fragmented.
8. Gate every route on adjacent shapes, padded strides, changed inputs, fresh output, and graph replay.

For M256 specifically, a useful first control is a 128-thread `BM64 x BN128 x BK128` design with
roughly 25 KiB LDS and one/two-stage variants. The target is structural parity with the tuned Triton
reuse pattern, not literal reproduction of Triton codegen.

## Verification ladder

Availability requires more than `import flydsl`:

```bash
export FLYDSL_GPU_ARCH=gfx1201
python -c 'from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())'
pytest -q tests/kernels/test_rdna4_wmma_atom.py
pytest -q tests/kernels/test_rdna_gemm.py
```

On the local R9700/gfx1201 checkout containing upstream commit `3c03e979`, the atom suite passed 6/6.
The applicable RDNA GEMM cases passed 26/26; 71 cases were skipped because they are gfx11-only. This
is physical gfx1201 validation. gfx1200 shares the gfx120x lowering but remains compile-supported,
not hardware-validated by this receipt.

For FP8 block scale, additionally require:

- reference parity over multiple random draws;
- every coarse route and its boundaries;
- odd padded B stride;
- explicit stream, graph replay, fresh output, and changed-input checks;
- a kernel trace proving one FlyDSL dispatch and no fallback;
- ISA containing the intended FP8 `v_wmma` and no scalar dot emulation;
- compiler resource evidence and a paired A/B timing receipt.

An import-only probe proves package presence. An MLIR-only test proves lowering. A device atom test
proves the instruction. Only the full ladder proves an operator route.

## On-box evidence snapshot

Hardware: Radeon AI PRO R9700, `gfx1201`, 64 physical CUs / 32 WGPs. The atom is now upstream at
`3c03e97919bedbeb95ea803baed089c3725eabb6`. The custom small-M operator prototype was originally
exercised on `eed78c6dd93fd297765632861587d9c3be82e0fc` and remains an uncommitted campaign snapshot,
so its hashes are recorded explicitly:

- `kernels/gemm/rdna4_fp8_blockscale.py` —
  `ea5eec5ee3a7d7d5b1bfa1081ef2fa82d25b4f25e139d709942ebfd209bcabd1`
- `kernels/common/gfx12_sync.py` —
  `5ceeff9d76d8181ca5279a904dca79c6c5c69a3a55bf4b4e07659e5cb50e3661`
- `tests/kernels/test_rdna4_fp8_blockscale.py` —
  `578fb0411b9f94a36c39713052092c9384a9cbe347b0c7df82f774e78744b4f8`
- `tests/perf/bench_rdna4_fp8_blockscale.py` —
  `12f384eaa36420840b14b7a7a2bcb1445f704403afc49408ca6e9243ba45c4b3`

The small-M kernel passed FP32-per-K128 parity, padded-stride, stream, fresh-output, and graph-replay
tests. Its representative kernel-only HIP/FlyDSL ratios ranged from 0.91x at M1 to 2.47x at a tuned
M39 long-K route, with most non-split routes above 1.0x. These are HIP comparisons for that prototype,
not evidence that broad-M FlyDSL beats Triton.

## Non-transferable CDNA advice

- MFMA names, wave64 geometry, AGPR tables, and XCD grid swizzles do not apply.
- gfx950's 160 KiB LDS budget does not apply.
- AITER-hosted FlyDSL kernels and their preshuffled weight layouts are not the RDNA4 integration seam.
- gfx950/gfx1250 scaled-MMA formats do not implement arbitrary FP32 K128 scales.
- A compile that recognizes `gfx1201` may still lack the Python or atom surface used by a newer kernel;
  compile the exact source before admitting it.

## Cross-links

- Deferred parity and campaign prerequisites: [`rdna4_campaign_readiness_backlog.md`](rdna4_campaign_readiness_backlog.md)
- Hardware model: `kernel_workflow/knowledge/amd_rdna4.md`
- Generic FlyDSL model: [`overview.md`](overview.md)
- GEMM workflow: [`authoring_gemm_levers.md`](authoring_gemm_levers.md)
- Debugging: [`debugging.md`](debugging.md)
- Dense operator: [`../../operators/dense_gemm/backends/flydsl.md`](../../operators/dense_gemm/backends/flydsl.md)
- Scaled quant operator: [`../../operators/scaled_quant_gemm/backends/flydsl.md`](../../operators/scaled_quant_gemm/backends/flydsl.md)
