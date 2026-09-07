---
id: flydsl_rdna4_fp8_blockscale_small_m
title: Port raw-weight FP8 K128 block-scale decode GEMM from HIP to direct FlyDSL on gfx1201
kind: expert_skill
authors: [chefjeff]
scope: kernel
match:
  operator:
    - dense_gemm
    - scaled_quant_gemm
  arch_class:
    - '*'
  gens:
    - gfx1201
  dtypes:
    - fp8_e4m3
  regimes:
    - decode
  from_backend: hip
  to_backend: flydsl
  profile_signature:
    op_name_regex: "rdna4_fp8_block_scaled_mm_decode|w8a8.*block.*scale"
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.05
  parity: required
validation:
  status: validated
  last_verified: '2026-08-27'
  gpu: 'Radeon AI PRO R9700 / gfx1201 (64 physical CUs, 32 WGPs)'
  model: 'Qwen3.8-27B-FP8 operator campaign'
  measured:
    isolated: 'kernel-only HIP/FlyDSL: M4 1.05-1.07x, M16 1.23x, M17 1.30x, M33 1.19x, M39 2.47x, M48 1.35x, M64 1.28x; split-K M2 1.03x and M1 0.91x remain HIP fallbacks'
    e2e_pct: ''
    parity: 'FP32-per-K128 oracle at rtol=0.02/atol=0.0625; all route families, odd padded B stride, current stream, fresh output, changed-input graph replay passed'
  artifact: skills/flydsl_rdna4_fp8_blockscale_small_m/validation_gfx1201.yaml
  notes: 'Validated for M1--M64 raw-weight decode/small-prefill routing. The skill promotes only measured FlyDSL-winning route families and keeps M1/M2 behind the incumbent unless a fresh paired receipt wins. It is not a broad-M prefill recipe.'
role: advisory_prior
supersedes: []
---

## When to use

Use for a gfx1201 raw-weight W8A8 FP8 E4M3FN GEMM with arbitrary FP32 A/B scales per K128,
BF16 output, and small M (validated M1--M64), when the incumbent is the native HIP decode kernel.
The tensor contract is A `[M,K]`, B `[N,K]` with contiguous K and possibly padded row stride,
A scale `[M,K/128]`, and B scale `[N/128,K/128]`.

This is a coarse route-family recipe, not an exact captured-shape table. gfx1200 shares the atom
family but was not in the validation receipt, so treat this skill as reference-only there. Read
`languages/flydsl/rdna4.md` before using it. Do not trigger it for broad-M prefill or for CDNA.

## Mechanism

gfx120x exposes wave32 `16x16x16` FP8 WMMA with eight packed A/B elements and eight FP32 accumulator
slots per lane. At small M, staging the tiny A operand through LDS adds more synchronization than
reuse. A register-fed kernel can instead:

- stream packed FP8 fragments directly into the WMMA atom;
- reuse B fragments across multiple live row fragments;
- batch independent VMEM loads before dependent WMMAs;
- accumulate exactly eight K16 WMMAs into one FP32 K128 partial;
- apply arbitrary FP32 A/B scales after the K128 dot, then accumulate and cast once;
- use split-K only for M1/M2 grid fill, with an LDS-only signal/wait reduction.

The measured gain is route-dependent. Non-split routes from M4 through M64 beat the native HIP
operator by 1.05--2.47x kernel-only. M1 did not win and M2 was marginal, so the validated integration
retains the HIP incumbent for those routes until a fresh receipt says otherwise.

## Procedure

1. Freeze the HIP callable and a FP32-per-K128 reference. Record M/N/K, B row stride, current stream,
   output allocation, and graph-capture behavior.
2. Build against standalone upstream FlyDSL at a revision containing the gfx120x FP8/BF8 atom from
   `ROCm/FlyDSL@3c03e979`. Set `FLYDSL_GPU_ARCH` to the detected gfx target. Never import
   `aiter.ops.flydsl`. The validation receipt below used the atom's pre-upstream fork commit; the
   custom operator itself must still be checked in before this recipe is clean-clone reproducible.
3. Implement one coarse router covering adjacent shapes:
   - M4/cache-resident: 64 threads, two waves, N64;
   - M5--M16: 128 threads, four waves, N128;
   - M17--M64: select row-tile count and N pairing from M/N/K regimes, not exact signatures;
   - keep M1/M2 on HIP by default; their FlyDSL split-K routes are research controls.
4. In every FlyDSL route, load raw B with its runtime row stride. Do not preshuffle or persist weights.
5. For each K128 group, initialize a fresh FP32 partial, issue eight K16 FP8 WMMAs, load one A scale
   per live row and one B scale per N128 group, update the FP32 total, and discard the partial.
6. Cache compilation/module selection outside the timed call. Launch on the current PyTorch stream and
   return a fresh output unless the caller supplies an explicit output tensor.
7. Gate every route family, both sides of each boundary, odd padded B stride, changed inputs, explicit
   stream, and graph replay at `rtol=0.02`, `atol=0.0625`.
8. Trace the candidate and inspect ISA. Require one FlyDSL operator dispatch, the intended FP8
   `v_wmma`, and no fallback or scalar-dot emulation.
9. Measure paired, order-reversed HIP/FlyDSL kernel-only and callable timings. Promote only route
   families above 1.05x; leave losing or marginal families on HIP.

## Knobs & pitfalls

- Search 64 versus 128 threads, one/two/four row tiles, N64 versus N128, and whether adjacent N
  fragments share a wave. Keep routing coarse.
- For M1, exposing the whole K128 batch means 40 VMEM loads followed by 32 WMMAs. This is register
  expensive and still measured 0.91x versus HIP; do not enable it from theory alone.
- For M2, two-way K split measured about 1.03x. That is below this skill's promotion floor.
- A split-K reduction may use `s_wait_dscnt(0)` plus `s_barrier_signal(-1)`/`s_barrier_wait(-1)` only
  when all waves execute the same instance. Divergence around the barrier can hang the GPU.
- `weight_scale[n//128, kb]` is shared by 128 output columns. Do not index it per 16-column WMMA.
- OCP E4M3FN is not CDNA3 FNUZ. gfx1250 atoms and scaled-WMMA shapes are also not gfx120x features.
- An import probe or MLIR test is insufficient. Require device parity, trace, ISA, and paired timing.

## Do-no-harm notes

- Keep M1 and M2 on the incumbent unless a same-session receipt clears 1.05x with parity.
- Do not apply this register-fed structure to M256/broad prefill. The missing cross-wave LDS reuse is
  exactly why the broad campaign failed to beat Triton.
- Never replace arbitrary FP32 K128 scales with E8M0/native scaled-WMMA semantics.
- Do not use AITER, CK, preshuffled or persistent weights, memoized activations/outputs, or extra device
  dispatches under this recipe.
- If the exact gfx120x FP8 atom cannot compile and emit `v_wmma`, mark the route unavailable and retain
  HIP; do not silently lower to a scalar fallback.

## Sources

- `ROCm/FlyDSL@3c03e97919bedbeb95ea803baed089c3725eabb6` — upstream gfx120x FP8/BF8 WMMA atom.
- `big-yellow-duck/FlyDSL@eed78c6dd93fd297765632861587d9c3be82e0fc` — historical pre-upstream
  atom revision used by the 2026-08-27 performance receipt.
- On-box kernel snapshot SHA-256
  `ea5eec5ee3a7d7d5b1bfa1081ef2fa82d25b4f25e139d709942ebfd209bcabd1`.
- On-box gfx12 synchronization helper SHA-256
  `5ceeff9d76d8181ca5279a904dca79c6c5c69a3a55bf4b4e07659e5cb50e3661`.
- [`validation_gfx1201.yaml`](validation_gfx1201.yaml) — per-route timings, parity scope, provenance,
  and non-trigger boundaries.
- [`../../../languages/flydsl/rdna4.md`](../../../languages/flydsl/rdna4.md) — complete architecture
  capability ledger and broad-prefill negative result.
