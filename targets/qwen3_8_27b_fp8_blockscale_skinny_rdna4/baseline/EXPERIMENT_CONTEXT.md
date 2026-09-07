# Native skinny FP8 campaign contract

Read this before proposing a direction. The seed is vLLM's validated gfx12x
hybrid: native HIP/rocWMMA for M=1,2, selected Triton BM32 prefill routes, and
generic Triton elsewhere. This target changes only the skinny M=1..16 region.

## Evidence

With `NCCL_PROTO=Simple`, `_w8a8_triton_block_scaled_mm` is the largest captured
kernel. All steady decode calls use M=4. Across 31 steps, the five shapes consume
24.93472 ms per step and 256 calls per step:

| M,N,K | Calls/step | Average us | GEMM share |
|---|---:|---:|---:|
| 4,17408,5120 | 64 | 163.47 | 40.63% |
| 4,5120,8704 | 64 | 107.82 | 26.80% |
| 4,8192,5120 | 48 | 75.62 | 14.10% |
| 4,5120,3072 | 64 | 44.33 | 11.02% |
| 4,7168,5120 | 16 | 69.08 | 4.29% |

The production source routes only M=1,2 into
`rdna4_fp8_block_scaled_mm_decode`; M=4 reaches generic Triton with BM64. The
native M=2 kernel's key advantage is packing both live rows into rocWMMA's
16-row A fragment so one B traversal serves both outputs. Generalize that
mechanism instead of serially invoking the M=1 kernel once per row.

## Immutable math and deployment contract

- gfx1200/gfx1201, wave32 rocWMMA FP8 E4M3 inputs and BF16 output.
- A is contiguous MxK. B is logical NxK with `stride(1)=1` and may have padded
  row stride. Activation scales are Mx(K/128); weight scales are
  (N/128)x(K/128), both contiguous FP32.
- N and K are positive multiples of 128. Every 128-wide K partial is
  accumulated and independently scaled in FP32; scaled partials are summed in
  FP32 and cast once to BF16.
- One device dispatch, no weight preshuffle/copy, no persistent result buffer,
  and no activation/output memoization. Results must reflect current inputs and
  own fresh storage.
- Candidate native routing must cover every integer M from 1 through 16 using
  a compact interval/family predicate. Exact Qwen `(M,N,K)` tables are invalid.
- Do not edit `/app/vllm` during the GEAK job. `binding.cpp` vendors the
  validated standalone M=1,2 source from the previous campaign; candidate HIP
  work belongs in the task workspace.

## Search priorities

1. Pack 4, 8, or 16 live rows into rocWMMA A fragments so each loaded B tile is
   reused across rows while maintaining independent activation scales.
2. Compare 64- and 128-column N tiles, two/four-wave workgroups, split-K versus
   no-split regimes, and special handling based on row-count buckets—not exact
   N/K shapes.
3. Balance grid parallelism against redundant B traversal. The R9700 exposes
   32 WGP scheduler units, so very wide N already provides ample workgroups.
4. Keep M=1 and the validated M=2 wide/small paths when a generalized kernel
   cannot match them; a small native family is acceptable if it covers the
   interval without fallback.

## Rejected shortcuts

- Calling the native M=1/M=2 operation repeatedly for additional rows adds
  dispatches and rereads B.
- Routing only M=4 is not comprehensive coverage.
- Leaving M=3,5..16 on Triton violates the engagement gate even if the weighted
  score improves.
- Stored outputs, pointer-key caches, synthetic constant inputs, and relaxed
  scaling order invalidate the result.

## Promotion

All 112 manifest cases require live parity. M=1,2 Qwen cases and every expansion
case have a 0.98x per-case floor. The counted M=4 ratio-of-sums must reach
1.01x. Confirm native engagement for every M, then repeat three paired
cache-flushed passes, CUDA-graph replay, and full-model serving with collective
settings fixed.
