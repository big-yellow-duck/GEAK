---
title: RDNA4 FlyDSL campaign-readiness backlog
kind: backlog
gens: [gfx1200, gfx1201]
dtypes: [fp8_e4m3]
regimes: [prefill, decode]
status: deferred
updated: 2026-08-28
---

# RDNA4 FlyDSL campaign-readiness backlog

This records the remaining work identified by the 2026-08-28 audit of
`feat/flydsl-rdna4-capabilities`. The existing RDNA4 material is sufficient for architecture-aware
agent reasoning and controlled gfx1201 small-M authoring. It is not yet equivalent to CDNA in
clean-checkout reproducibility, validated recipe breadth, or broad-M performance evidence.

Do not interpret this backlog as a ban on FlyDSL. Its purpose is to make the prerequisites for a
competitive RDNA4 FP8 GEAK campaign explicit and testable.

## Audit verdict

| Dimension | State | Evidence / consequence |
|---|---|---|
| Architecture model | pass | wave32, WGP versus CU, gfx120x v8 WMMA fragments, LDS limits, OCP FP8, and non-transferable CDNA assumptions are documented |
| Agent routing | pass | planning and author roles require `languages/flydsl/rdna4.md` for gfx1200/gfx1201 FlyDSL work |
| FP8 K128 numerical contract | pass | arbitrary FP32 A/B scales are applied to completed K128 FP32 partials before one BF16 cast |
| Small-M gfx1201 recipe | pass, narrow | one validated M1--M64 HIP-to-FlyDSL recipe; M1/M2 retain HIP fallback |
| Broad-M / M256 recipe | open | no tracked LDS-tiled FlyDSL implementation or validated Triton-beating receipt |
| Reproducible source | blocked | the raw-weight block-scale operator, sync helper, tests, and benchmark are an uncommitted on-box snapshot |
| Bootstrap | blocked | `ensure_flydsl` matches gfx942/gfx950 and builds a CDNA pin rather than the gfx120x FP8 fork |
| Capability breadth | not at CDNA parity | CDNA has 14 indexed FlyDSL operator families and six validated performance recipes; RDNA4 has two families and one gfx1201 recipe |
| Machine metadata | needs correction | combined generation/dtype lists can form false cross-products and globally mark experimental RDNA routes as `sota` |
| Measurement provenance | partial | small-M route timings exist, but broad-M ISA/resource/trace receipts are not stored as a reproducible artifact bundle |

## P0 — finish before launching a competitive GEAK campaign

### 1. Make the FlyDSL fork self-contained

- Commit and push `kernels/gemm/rdna4_fp8_blockscale.py`.
- Commit and push `kernels/common/gfx12_sync.py`.
- Commit and push the parity/router tests and performance benchmark.
- Preserve the recorded snapshot hashes in the commit or migration note and confirm the committed
  files reproduce them, or document every intentional delta.
- Pin GEAK to the resulting immutable FlyDSL commit rather than to an on-box working tree.

Acceptance: a fresh clone contains every source named by the small-M skill and runs its complete
verification ladder without copying files from the R9700 container.

### 2. Add an exact RDNA4 bootstrap and preflight

- Add a gfx1200/gfx1201 dependency skill or extend `ensure_flydsl` with an architecture-specific pin.
- Do not admit an arbitrary installed package solely because its version is at least `0.2.2`.
- Verify the exact gfx120x FP8 atom/API by compiling and running the device atom test.
- Run operator parity, padded-stride, explicit-stream, fresh-output, changed-input, and graph-replay
  checks before exposing FlyDSL to GEAK agents.
- Inspect emitted ISA for `v_wmma_f32_16x16x16_fp8_fp8` and reject scalar/emulated FP8 dot paths.

Acceptance: the bootstrap succeeds from a clean ROCm container, writes a reusable environment file,
and proves the exact fork rather than only `import flydsl`.

### 3. Split machine-readable CDNA and RDNA4 capability records

- Avoid one frontmatter record whose `gens` and `dtypes` imply a Cartesian product.
- Advertise gfx120x only for OCP `fp8_e4m3` and the formats actually compiled on that target.
- Keep FNUZ, MXFP4/FP4, MFMA, AITER, and CDNA preshuffle claims on CDNA-only records.
- Mark broad-M RDNA4 FlyDSL `experimental` until a reproducible winner exists; do not inherit global
  `sota` status from the CDNA implementation.
- Ensure generated capability/SOTA source lists include the RDNA4 source commit rather than only
  CDNA AITER files.

Acceptance: a machine query for gfx1201 + FP8 returns the direct RDNA path and correct sources, while
queries for gfx1201 + FNUZ/FP4 do not select unsupported FlyDSL routes.

### 4. Classify current upstream RDNA FP8 material

- Add upstream `kernels/gemm/rdna_fp8_preshuffle_gemm.py` as a small-M scheduling reference.
- State explicitly that its per-token/per-channel, preshuffled-weight contract is not the vLLM raw-B,
  arbitrary-FP32 K128 block-scale contract.
- Reconcile the low-level direct `rocdl.wmma` FP8 path with the fork's higher-level gfx120x FP8 atom
  extension so agents choose the intended API deliberately.

Acceptance: agents can reuse current upstream load/scheduling ideas without silently changing the
operator layout or scale semantics.

## P1 — evidence needed for broad-M parity

- Check in one correct LDS-tiled M256 seed, initially comparing single- and two-stage pipelines.
- Record paired order-reversed timings, a cache-flushed control, and a valid
  `GEAK_TIMING_RECEIPT` from the current dispatch-priming harness.
- Store generated ISA, VGPR/SGPR/LDS/scratch metadata, and one engagement trace with the receipt.
- Validate all manifest boundaries, both N/K families, padded and contiguous B, five random draws,
  output independence, current-input behavior, graph replay, and exactly one device dispatch.
- Require greater than 1.01x on the scored M256 case and at least 0.98x on every retention gate before
  promotion.
- Add a broad-M expert skill only after the tracked implementation passes those gates. Until then,
  keep the architecture card as guidance and do not label a speculative recipe validated.
- Add a gfx1200 receipt or explicitly retain gfx1200 as compile-only/experimental; current device
  evidence is gfx1201-only.

## Native-HIP M256 campaign evidence

Run: `team_baseline_20260828_043431_1519294_8385/baseline` on gfx1201. The five-round campaign
finished without a deadline hit and promoted no patch, so the frozen Triton implementation remains
the incumbent.

- Round 1 produced a correct BM64xBN128xBK128, four-wave32, single-stage FP8-WMMA kernel, but maximal
  VGPR pressure, private storage, and feed cost left it at 0.772136x on the scored case.
- Round 2 reported a locally attractive 1.264365x scored result from a two-stage 25,088-byte LDS
  design, but independent correctness failed. This is a structural lead, not performance evidence.
- Round 3 removed scratch/spills and proved real FP8 WMMA, but reached only 0.839316x scored; the
  alternate N5120/K8704 gate fell to 0.608277x.
- Round 4's BM96, six-wave design improved the M257/M272 boundary cases to 1.089153x/1.106464x, but
  the scored case remained 0.905079x and long-K remained weak.
- Round 5 recovered long-K to 0.938243x with branchless safe prefetch and a spill-free A3/B7
  schedule. The scored result was 0.907827x, with 254 VGPRs, 34 SGPRs, 32 KiB LDS, zero private
  segment, zero spills, and native FP8 `v_wmma` present.
- The final director correctly retained the byte-identical Triton source. Its no-op validation was
  flagged because the frozen benchmark emitted no `GEAK_TIMING_RECEIPT`; re-freezing the harness is
  mandatory before treating its 0.990153x remeasurement as authoritative.
- Budget accounting needs repair or clarification: the workflow summary reports 17/16 units while
  the Tech Lead report describes 9/16. Five rounds were actually issued.

The main lesson is that native WMMA engagement and zero spills are necessary but insufficient.
Operand-feed efficiency, VGPR occupancy, boundary geometry, and long-K reuse remain the decisive
gaps to Triton's generated kernel. The round-2 two-stage structure is worth reconstructing only after
its correctness failure is isolated.

## Done definition

The RDNA4 FlyDSL prerequisite is complete when all of the following hold:

- every cited source and test is committed and pinned;
- a fresh-container bootstrap proves the exact gfx120x FP8 path;
- capability metadata cannot cross-match CDNA-only formats or maturity claims;
- the upstream RDNA FP8 reference is classified without changing the target contract;
- a receipt-bearing, correct M256 FlyDSL kernel beats Triton by more than 1.01x while every hard gate
  remains at least 0.98x;
- ISA, resource, trace, parity, stream, graph, and one-dispatch evidence is stored with the result.

## Sources

- `languages/flydsl/rdna4.md`
- `expert_skills/skills/flydsl_rdna4_fp8_blockscale_small_m/skill.md`
- `expert_skills/skills/flydsl_rdna4_fp8_blockscale_small_m/validation_gfx1201.yaml`
- `expert_skills/skills/ensure_flydsl/skill.md`
- `index/capability_index.yaml`
- On-box HIP run: `/app/GEAK/exp/team_baseline_20260828_043431_1519294_8385/baseline`
- https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
- https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_tuning_guide.md
- https://github.com/ROCm/FlyDSL/blob/main/kernels/gemm/rdna_fp8_preshuffle_gemm.py
- https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
