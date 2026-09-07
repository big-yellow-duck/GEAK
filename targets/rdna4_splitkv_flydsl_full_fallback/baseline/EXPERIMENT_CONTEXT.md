# RDNA4 FlyDSL full SplitKV-fallback campaign contract

Read this file completely before planning any optimization direction. The seed
is the validated standalone FlyDSL grouped-WMMA SplitKV kernel for its current
safe route, with vLLM's live Triton SplitKV implementation retained everywhere
else. The campaign goal is to replace that remaining Triton region with a small,
explainable family of faster FlyDSL kernels.

## Frozen source provenance

- FlyDSL checkout: `/app/FlyDSL`
- FlyDSL commit: `e649c87da2afb9728b576e8103c735e00810b972`
- Branch: `codex/rdna-dynamic-buffer-bounds-check`
- Attention launcher SHA-256: `d6ae8de377ee97f048f10ad2b24abe6adc1b4c9db5214e06ccfc438d1661835e`
- Grouped WMMA stage SHA-256: `f97954e94dde81a29e6668bc0638a5d64d94c8b033acf3f315506d511623a2bb`
- Tensor launcher SHA-256: `47bf4f068250002e1275956841b7441f46ca89a7cb761178f0544e4d5fec0060`
- vLLM checkout: `/app/vllm-rdna4-fp8-flydsl-tp2-hip-ar`
- vLLM commit: `57d49ecf01c5821461e63c4c5b122d86ed42e9db`
- Triton/dispatch source SHA-256: `84bcaf3ba33f7a87f20bda1de267081b1b390225c5698ac2f8b4eca87cffcf31`

These hashes include the current uncommitted integration state. Do not modify
either external checkout during the GEAK run. Candidate work belongs under
`kernel_src/`: add local FlyDSL modules and route to them from `kernel.py`.

## Operation contract

- Hardware: Radeon AI PRO R9700, gfx1201, RDNA4, wave32, 64 physical CUs,
  32 WGP scheduling/resource units.
- Decode query/output dtype: BF16 or FP16.
- KV cache: matching BF16/FP16, or per-tensor FP8 E4M3FN/E4M3FNUZ with
  independent scalar FP32 K/V scales.
- Head dimension: 128 or 256.
- GQA ratio: every integer from 1 through 16; positive KV-head count and exact
  divisibility are required.
- Paged K layout: `[blocks, kv_heads, D/x, page, x]`, where `x=16/itemsize`.
  V layout: `[blocks, kv_heads, D, page]`. The innermost dimensions are
  contiguous, but block strides are deliberately padded.
- Physical page size is runtime geometry and is not required to divide the
  logical compute tile. Campaign pages include 16, 32, 128, 544, 1056, 1568.
- Batch may be one or ragged. `query_start_loc` describes one decode token for
  every represented sequence.
- Split count is one of 2, 4, 8, 16. Time the complete stage-1 plus reduction
  operation using caller-owned output and FP32 scratch. Allocation, validation,
  and JIT compilation remain outside timing.
- Online softmax and cross-split composition use FP32 state. Apply FP8 scales
  before BF16/FP16 dot consumption. Cast the final result once to query dtype.
- Output must use the caller's fresh storage and reflect current inputs.

The immutable correctness truth is the live Triton SplitKV path, not a naive
PyTorch attention. It catches the exact serving arithmetic and page/stride ABI.

## Seed and prior evidence

The current seed routes only batch-one BF16-query, FP8-E4M3FN, D=256, GQA 6/7
to FlyDSL. Everything else remains on Triton.

The final paired sweep measured:

- GQA 6, 4014 tokens: 27.71 us FlyDSL vs 68.27 us Triton, 2.46x,
  max absolute error 0.000488.
- GQA 7, 8192 tokens: 43.03 us FlyDSL vs 111.94 us Triton, 2.60x,
  max absolute error 0.000488.

Do not blindly widen the existing route. On adversarial unnormalized inputs,
the current compiled GQA 1/4/5 variants were fast but differed from Triton by
roughly 0.06-0.09. Compile-time GQA 8 also exposed row corruption. A runtime
GQA masking repair was accurate but cost almost 2x, so it was not retained.
Generalized BF16 performance ranged from 0.69x to 1.55x, and ragged batches
were not consistently faster.

## Manifest and scoring

`cases.json` contains 66 cases:

- all GQA ratios 1-16 for BF16-query/FP8-cache/D256;
- the BF16, FP16, E4M3FN, and E4M3FNUZ cache/query/head-size cross-section;
- batch-one and ragged B3/B8 cases with one, two, or four KV heads;
- page/tail boundaries and arbitrary softmax scales;
- two current FlyDSL routes as zero-weight regression guards.

The 58 Triton-owned cases have equal manifest weight and form the primary
ratio-of-sums. Every case is also an individual correctness and >=0.98x
performance gate. The eight zero-weight seed/boundary cases cannot be traded
away for an aggregate win. Final replacement coverage must be reported as the
number of scored cases with proven FlyDSL source/trace engagement; a fallback
result is correct but is not progress toward replacement.

Never route by exact case name, exact sequence length, exact captured tuple, or
model identity. Coarse families based on dtype, D, GQA bucket, batch/grid
occupancy, page geometry, and context tiles are valid when neighboring cases
prove the rule.

## Productive 32-direction search space

1. Repair the exact-row epilogue for GQA 1-5 and design a correct GQA-8/16
   family. Prefer compile-time GQA buckets over expensive fully runtime masks.
2. Separate efficient GQA buckets such as 1, 2-4, 5-7, 8, and 9-16; tune how
   eight waves share query rows without corrupting inactive rows.
3. Add D=128 WMMA tiling without retaining dead D=256 work or workspace traffic.
4. Template the load/dequant path for E4M3FN, E4M3FNUZ, BF16, and FP16. Preserve
   the scale-before-dot contract for FP8.
5. Improve ragged scheduling: grid over active `(sequence, kv_head, split)`
   items and avoid making the shortest sequence pay for the longest.
6. Tune 32/64/128-token logical tiles independently of physical page size.
7. Reduce LDS traffic and bank conflicts in Q/K, score/probability, and V/P
   phase reuse; preserve required signal/wait ordering.
8. Pipeline page-table/K/V loads across the WMMA work and prefetch only valid
   addresses. Use the dynamic buffer-bounds support from the pinned branch.
9. Reduce split scratch and second-dispatch cost; consider fused or persistent
   reduction only after complete-operation timing proves the gain.
10. Inspect generated gfx1201 ISA, VGPR/SGPR/LDS, occupancy, and ATT stalls for
    every retained family. A source edit without engagement proof is not a win.

Use stable `fx.*`, tiled-copy, tiled-MMA, `SharedAllocator`, and `_run_compiled`
APIs. Do not introduce direct upstream MLIR dialect operations, AITER, CK,
CDNA MFMA assembly, hidden host precomputation, output memoization, persistent
operand copies, relaxed accuracy, or a modified Triton reference.

## Promotion gates

- Five fresh live-Triton comparisons per case at `rtol=atol=0.01`.
- Output storage independence and current-input dependence.
- Every case correct; every zero-weight or retained route at least 0.98x.
- Bank only reproducible improvements of at least 0.5%.
- Primary weighted speedup greater than 1.0x, with at least one newly engaged
  general FlyDSL bucket per banked round.
- Final goal: all 58 scored Triton cases use proven FlyDSL kernels and the
  weighted result beats the frozen hybrid seed. If full replacement is not yet
  reached, report exact remaining fallback families rather than hiding them.
- Final verification uses cache-controlled paired order reversal, batched CUDA
  event timing, CUDA graph replay, and representative ATT/ISA receipts.
