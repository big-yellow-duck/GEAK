# M256 native-HIP campaign contract

Read this file before planning or editing. It is the campaign memory and is mandatory.

## Objective and denominator

Beat the immutable vLLM Triton `w8a8_triton_block_scaled_mm` implementation on the primary
`M=256, N=8192, K=5120` case on the Radeon AI PRO R9700 (`gfx1201`) with a native HIP kernel.
The primary B tensor is logically `[8192, 5120]` but has row stride 5376. The one scored case has
positive weight; the nearby cases are zero-weight correctness and generality gates.

The archived GEAK wrapper median for the primary Triton case was 0.163797 ms. This target's clean
pre-launch measurement was 0.160477 ms (20 warmups, 100 timed repeats). A separate direct hot-cache
profile measured 0.1412 ms (101 samples; cold 0.2663 ms). Re-measure with the exact promotion
protocol before drawing conclusions: these values are orientation, not a substitute for paired A/B.

## Evidence from the Triton baseline

The live Triton route selected `BM=64, BN=128, BK=128, GROUP_M=32`, four waves and two stages. At
M256 this launches 256 output CTAs. The compiled kernel used about 25,088 bytes dynamic LDS, 234
VGPRs, 48 SGPRs, and no scratch. Its important advantage is structural: four wave32s cooperate on a
64x128 output tile, stage contiguous FP8 A/B tiles in LDS, reuse each staged tile across four output
waves, overlap the next K tile, and keep the scaled K128 partials in registers before one BF16 store.

This is a compute-heavy GEMM, not a simple bandwidth kernel. A native HIP candidate must preserve
operand reuse and produce `v_wmma*` in ISA; scalar/vector FP8 emulation cannot plausibly win.

## Exact operator contract

- `A` is FP8 E4M3FN `[M,K]`, contiguous.
- `B` is FP8 E4M3FN `[N,K]`, `stride(1)=1`, with either padded or contiguous `stride(0)`.
- `A_scale` is FP32 `[M,K/128]`, contiguous.
- `B_scale` is FP32 `[N/128,K/128]`, contiguous.
- For every K128 block, complete the FP8 dot in FP32, multiply by that block's independent FP32 A
  and B scales, accumulate the scaled partial in FP32, and cast once to BF16.
- Return a fresh BF16 `[M,N]` tensor that reflects current input values. The call must be graph-safe.
- Correctness is five live random draws at `rtol=0.02`, `atol=0.0625`, plus mutation and fresh-output
  checks.

## Native HIP requirement

Accepted work must execute one native HIP device route for every manifest case. Candidate code may
not call Triton, FlyDSL, AITER, CK/CK-Tile, rocBLAS, hipBLASLt, or another library GEMM. The baseline
keeps Triton only as the immutable denominator and reference. Do not use generated Triton as a HIP
substitute.

Compile explicitly for gfx1201 and use the current PyTorch HIP stream. Cache compilation/module
loading outside the timed call; never synchronize the host inside `run`. One timed invocation must
issue exactly one GEMM dispatch. No weight preshuffle, persistent weight copies, input/output
memoization, data-dependent caches, or auxiliary reduction dispatches are allowed.

Do not key a fast path to the exact signature or model identity. One explainable M240--M272 route
must cover the primary, M tails, both padded and contiguous B rows, and the second N/K family. A
failed gate closes that design, not the whole campaign.

## Starting HIP design and search axes

Begin with a correct wave32 WMMA design rather than a scalar kernel:

- 128 threads (four wave32s), BM64 x BN128 x BK128, matching the baseline's 256-CTA grid.
- Cooperative aligned 128-bit global loads into roughly 25 KiB LDS, with padding chosen from measured
  bank behavior. Respect the portable 64 KiB per-workgroup cap.
- Reuse the same A/B LDS tile across all four output waves. Start with a correct single-stage loop,
  then add a measured two-stage/prefetch schedule.
- Use rocWMMA if it generates the needed gfx12 FP8 WMMA path; raw gfx12 builtins are allowed when
  fragment layout control is necessary. Inspect ISA for `v_wmma*` after every structural rewrite.
- Preserve K128 scale boundaries exactly. Overlap/coarsen scale loads only if reduction order and
  parity remain valid.
- Compare BM/BN geometry, 64/128/256 threads, LDS padding/load mapping, prefetch distance, accumulator
  layout, vector BF16 stores, and WGP fill. Recompile and record VGPR/SGPR/LDS/scratch for real winners.

Avoid spending the campaign on cosmetic flags. A candidate must first prove that its HIP source was
rebuilt, its dispatch appeared in a trace, and the intended WMMA instructions appeared in ISA.

## Promotion gates

- Use the entire 16-direction budget unless correctness or infrastructure truly blocks progress.
- Bank only reproducible gains of at least 0.5% against the same frozen Triton oracle.
- All seven cases must pass; all zero-weight gates must stay at least 0.98x.
- Target greater than 1.01x on the primary scored case.
- Before final promotion, use paired order-reversed timing, a cache-flushed check, graph replay, HIP
  dispatch trace, and ISA/resource evidence. Report failures honestly; never redefine the denominator.
