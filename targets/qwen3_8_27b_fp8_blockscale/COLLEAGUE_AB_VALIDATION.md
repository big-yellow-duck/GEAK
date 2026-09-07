# RDNA4 scaled-mm A/B validation

Date: 2026-08-24

## Decision

Keep the native-HIP/BM32 implementation from `rdna4-fp8-blockscale` as the
scaled-mm foundation. Do not replace it with the colleague branch's tuned
Triton configurations or fused M=1 kernel wholesale.

The colleague attention and custom-all-reduce work is independent of this
scaled-mm result and needs its own serving-level evaluation.

## Compared revisions

- Common base: `7ca49fbe4b`
- Ours: `fd381955ad9c402c0173eebaa42a738b091bd410`
- Colleague: `e8370f66ad143a7fc8c5124bf3a98b9f1add52e4`
- Colleague worktree: `/app/vllm-colleague`
- Device: Radeon AI PRO R9700 (`gfx1201`)

The colleague checkout supplied vLLM and its Triton kernels. The benchmark
loaded our `rdna4.py` and built `_rocm_C` into the same process, keeping the
Python, PyTorch, Triton, tensors, and timing protocol identical.

## Protocol

- All 45 frozen Qwen3.8-27B-FP8 TP2 tensor/layout signatures
- Original padded strides and captured call-count weights
- Correctness checked before timing
- 10 warmups and 51 timed samples
- 512 MiB cache eviction before every timed sample
- CUDA-event device timing, one operation per sample
- Both measurement orders, with the median of the two leg medians
- Supplemental captured-graph decode run with the same cold-cache protocol
- Supplemental runs on every exact shape configured by the colleague branch

All eager legs used `cuda_event`; all graph legs successfully used
`cuda_event_graph` without falling back to eager timing.

## Exact 45-shape result

| Regime | Cases | Our wins | Colleague wins | Weighted speedup from ours |
|---|---:|---:|---:|---:|
| Decode M=1/2 | 10 | 10 | 0 | **1.800x** |
| Prefill | 35 | 18 | 17 | **1.028x** |
| All | 45 | 28 | 17 | **1.111x** |

The raw win count understates the prefill result because the implementations
are identical on 18 fallback cases. Small apparent wins on those cases are
timing noise. Split by actual route:

| Prefill route | Cases | Weighted speedup from ours |
|---|---:|---:|
| Our BM32 specialization | 17 | **1.072x** |
| Shared fallback | 18 | 0.991x |

BM32 won every routed case. Its weighted speedup by M was 1.056x at M=72,
1.092x at M=249, 1.101x on the four routed M=523 shapes, and 1.023x on the
three routed M=784 shapes.

The colleague branch has no exact `(N,K)` configuration match in the frozen
TP2 workload. Its exact-45 leg therefore represents its preserved baseline
fallback. Its six tuned shapes are Qwen3.8 TP4-sized projection shards.

## End-to-end decode

This leg starts with BF16 activation input and includes activation
quantization plus GEMM.

| Context | Cases | Our wins | Weighted speedup from ours |
|---|---:|---:|---:|
| Eager, exact TP2 decode | 10 | 10 | **1.913x** |
| Captured graph, exact TP2 decode | 10 | 10 | **1.624x** |

The colleague fused kernel is not selected on these TP2 shapes, because its
configuration files do not match them.

## Colleague branch's intended decode routes

The quantized-GEMM comparison exercised M=1 and M=2 for all six tuned `(N,K)`
shapes. Our native HIP kernel won all 12:

| Context | Our wins | Aggregate speedup from ours |
|---|---:|---:|
| Eager | 12/12 | **1.864x** |
| Captured graph | 12/12 | **1.595x** |

After relaxing the HIP kernel from K%256 to K%128, our separate quantization
plus native HIP GEMM won all five fused BF16-input M=1 shapes:

| N | K | Eager speedup from ours | Graph speedup from ours | Winner |
|---:|---:|---:|---:|---|
| 2048 | 5120 | 2.028x | 1.671x | Ours |
| 4352 | 5120 | 1.938x | 1.724x | Ours |
| 5120 | 2176 | 1.762x | 1.396x | Ours |
| 5120 | 4352 | 1.667x | 1.481x | Ours |
| 5120 | 768 | 1.676x | 1.023x | Ours |

The five-shape aggregate is 1.819x eager and 1.526x under captured graphs.
For K=2176, the 17 scale blocks are divided 9+8 between split waves instead of
truncating them to 8+8. This adds no padding, allocation, copy, or kernel
launch. Focused tests also cover K=128, 256, 384, 640, and 2176 for M=1/M=2.

## Correctness

- Exact quantized maximum absolute difference: `7.63e-06`
- Exact decode end-to-end maximum absolute difference: `9.77e-04`
- Intended quantized maximum absolute difference: `3.81e-06`
- Intended fused maximum absolute difference: `7.81e-03`

All comparisons passed their configured BF16/FP8 tolerances.

## Integration recommendation

1. Keep our RDNA4 scaled-mm registration, HIP M=1/M=2 kernel, BM32 prefill
   specialization, and current routing.
2. Do not merge the six ordinary Triton decode config files; the HIP kernel is
   faster on every configured M=1/M=2 case.
3. Do not enable the fused BF16-input kernel; the K%128 HIP route now wins all
   five of its configured shapes in eager and graph timing.
4. Evaluate the colleague partitioned paged-attention kernel independently;
   it does not overlap scaled-mm and may still be worth merging.
5. Treat the AITER custom-all-reduce changes separately. They are TP4-specific
   and the published branch still depends on AITER/platform support not
   exercised by this TP2 scaled-mm benchmark.

## Artifacts

- `benchmark_colleague_ab.py`: reproducible A/B driver
- `colleague_ab_results.json`: complete 45-shape eager results
- `colleague_ab_decode_graph_results.json`: captured-graph decode and intended-route results
- `colleague_ab_k128_decode_results.json`: eager results after K%128 support
- `colleague_ab_k128_decode_graph_results.json`: graph results after K%128 support
