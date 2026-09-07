# RDNA4 Triton/rocWMMA validation verdict

## Final backend verdict

The cumulative GEAK Triton-lane implementation with round-4 engineer 1's wide M=2 decode path is
**the selected winner for the current vLLM gfx1201 integration**. It passes the captured mathematical
and layout contract, produces independent outputs, reproduces its principal decode speedup, and
improves the captured decode+prefill workload by approximately 1.10x in fresh paired measurement.

The selection is:

```text
WINNER: Triton + HIP/rocWMMA shape-routed hybrid
STATUS: kernel-ready for vLLM integration on gfx1201
FLYDSL: defer and revisit after deployment-layout work
```

This is a kernel-readiness verdict, not a claim that the current experiment directory can be copied
unchanged into vLLM. Upstream integration still needs production packaging of the HIP/rocWMMA source,
normal vLLM dispatch/configuration, CI coverage, and an end-to-end offline serving run. The present
Python wrapper uses `torch.utils.cpp_extension.load_inline`, which is appropriate for the isolated
experiment but should not be the final upstream build path.

This selection does not imply that FlyDSL is unsuitable for RDNA4. FlyDSL is still young, its native
gfx1201 WMMA path works, and its clean round-4 candidate showed real gains. The current Triton/rocWMMA
hybrid wins because it has the stronger reproduced workload result and a practical fallback/deployment
path today.

## FlyDSL review and deferral

FlyDSL round 3 is disqualified. Its reported 65.34x geomean and 81.37x weighted speedup came from an
activation-dependent result cache keyed by the four input tensor objects and their versions. Repeat
benchmark calls returned a previously computed GEMM result, sometimes reusing already-filled allocator
storage without a copy. Passing randomized parity and fresh-output-storage checks did not make that a
valid kernel result: the timed cache-hit path did not recompute the GEMM from current activations.

FlyDSL round 4 removed that result replay. Its clean source allocates a fresh output and launches the
FlyDSL GEMM on every call; only an immutable-weight-derived preshuffle is cached. Fresh cache-hit
remeasurement passed correctness and produced approximately:

| Round-4 FlyDSL regime | Fresh speedup |
|---|---:|
| Decode | 1.111x |
| Prefill | 1.060x |
| Captured combined workload | 1.069x |
| Unweighted geomean | 1.062x |

The current wrapper retains only two preshuffled weights. Qwen traverses many distinct layer weights,
so those entries will normally be evicted before the next token revisits a layer. Forced preshuffle
cache misses measured only 0.41-0.50x of the vLLM baseline on representative decode and wide-prefill
cases. Recovering the cache-hit result in a real server therefore requires a production weight-layout
strategy: ideally preshuffle once while loading and retain weights directly in the FlyDSL layout, or
otherwise prove that the memory cost and lifetime of transformed copies fit the deployment.

The final choice is consequently based on deployable evidence:

| Candidate | Valid recomputation | Fresh workload result | Current deployment status |
|---|---|---:|---|
| Triton + HIP/rocWMMA hybrid | Yes | ~1.103x | Selected; incumbent fallback and no full-model preshuffle dependency |
| FlyDSL round 4 | Yes | ~1.069x on weight-cache hits | Deferred; weight layout/cache misses unresolved |
| FlyDSL round 3 | No | Disqualified | Activation-dependent result memoization |

FlyDSL should be revisited after its model-load weight-layout integration is implemented and the same
offline workload is rerun without relying on a two-entry benchmark-local cache.

## Candidate reviewed

- Cumulative routed source:
  `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_1/workspace/kernel_src/`
- Round-4 patch:
  `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_1/best_patch.diff`
- Round-4 engineer report:
  `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_1/report.md`
- Frozen workload: 45 exact tensor/layout signatures captured from Qwen3.8-27B-FP8 with TP=2 on a
  Radeon AI PRO R9700 (`gfx1201`, wave32, 64 physical CUs / 32 WGP scheduling units).

Round-4 engineer 1 changes only `binding.cpp`. It adds decode-specific rocWMMA implementations and
does not alter the Python prefill kernel or its routing.

## Correctness and contract

Fresh validation passed:

- all 45 captured shapes;
- five randomized input draws per shape;
- BF16 output parity at the frozen `rtol=0.01`, `atol=0.01` contract;
- fresh-output independence;
- captured padded B row strides;
- independently scaled FP32 partial accumulation for every 128-wide K block;
- gfx1201 wave32 execution using native FP8 WMMA rather than CDNA MFMA; and
- one logical device dispatch on the specialized decode paths.

The reviewed workload begins after dynamic activation quantization. Bias, output reshape, attention,
sampling, TP communication, and other model operations are not part of this isolated operator result.

## Timing protocol

The fresh audit ran with no active KFD workload and acquired a GEAK GPU lock before every measurement.
It used identical tensors for each A/B pair, CUDA-event device timing, explicit warmup and device
synchronization, and a 512 MiB cache flush outside every timed sample. Full-workload measurements used
10 warmups and 51 samples per implementation and were repeated in both baseline-first and
candidate-first order. The principal wide-decode case was also stressed with 20 warmups and 101
samples.

The preserved harness labels these receipts `timer_unprimed` because its result schema does not emit a
`primed` field. This audit did prime every implementation explicitly. The missing field remains a
reporting defect to fix before producing an upstream-facing automated receipt; it is not an absence of
warmup in these fresh measurements.

## Performance result

### Wide M=2 decode target

| Measurement | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| Original round-4 receipt | 0.492426 ms | 0.298443 ms | 1.64998x |
| Fresh full-suite A/B | 0.491265 ms | 0.295163 ms | 1.66439x |
| Fresh 101-sample stress run | 0.482915 ms | 0.300843 ms | 1.60521x |

Across the repeated 51-sample passes, the target stabilized at approximately 0.2952 ms and
1.65-1.66x. The round-4 target claim is reproduced.

### Captured decode+prefill mixture

The authoritative workload projection is a ratio of counted times:

```text
fresh speedup = sum(count * fresh baseline latency)
                -------------------------------------
                sum(count * fresh candidate latency)
```

| Aggregate | Time |
|---|---:|
| Fresh baseline | 912.86997 ms |
| Fresh candidate | 827.56848 ms |
| Fresh paired speedup | 1.10307x |
| Frozen baseline | 917.17192 ms |
| Frozen-baseline / fresh-candidate speedup | 1.10827x |

The reverse-order run measured 1.10360x from fresh counted times. The original independent
round-selection receipt was 1.11078x weighted, so the claimed aggregate improvement reproduces within
normal run-to-run variance.

These totals project only this FP8 block-scaled linear operator over the captured call frequencies.
They do not predict the complete vLLM request speedup without measuring the fraction of end-to-end
time spent in this operator.

### Prefill-only result

Prefill does not regress in fresh paired aggregate timing:

| Order | Baseline prefill | Candidate prefill | Speedup |
|---|---:|---:|---:|
| Baseline first | 775.470 ms | 761.401 ms | 1.01848x |
| Candidate first | 774.499 ms | 761.076 ms | 1.01764x |

The approximately 1.8% gain is concentrated in seven inherited BM32xBN128 workload signatures. They
produce five distinct constexpr `(M, N)` launch shapes because `K` remains a runtime argument. The
remaining 28 prefill signatures call the original vLLM Triton implementation, so small paired
fluctuations on those fallback cases are measurement noise rather than a different kernel
implementation.

## Deployment routing

The validated source is already a shape-routed hybrid:

```text
M=1 or M=2 decode
  -> gfx1201 HIP/rocWMMA specialization

M=523 with (N,K) in {(5120,3072), (5120,8704), (7168,5120), (8192,5120)}
M=784 with (N,K) in {(5120,3072), (5120,8704), (8192,5120)}
  -> specialized Triton BM32xBN128 prefill

all remaining prefill shapes and unsupported specialization conditions
  -> original vLLM Triton block-scaled GEMM
```

This routing preserves the large decode gain, retains the independently reproduced prefill wins, and
uses the incumbent implementation elsewhere. A decode-only integration would also be correct but would
discard the measured prefill improvement without reducing risk on the already-fallback shapes.

### Routing generalization screen

The follow-up crossover driver [`sweep_prefill_bm32.py`](sweep_prefill_bm32.py) bypasses the existing
router and launches BM32 directly against the incumbent BM64 kernel. It preserves captured padded
weight strides, checks output parity, uses cold-cache CUDA-event timing, measures both execution
orders, and summarizes the worst result across all five captured `(N, K)` weight shapes for each
sampled `M`.

The initial 5-warmup/11-sample sweep rejects one broad exclusive `32 < M < 784` rule. Representative
worst-shape results were 0.912x at `M=138`, 0.965x at `M=277`, 0.971x at `M=512`, and 0.947x at
`M=768`. This is not a monotonic crossover: changing BM64 to BM32 changes program count and CU
occupancy, and the outcome also depends on `N` and `K`.

Two narrower findings are promising:

- sampled `64 <= M <= 128` points were non-regressing across all five weight shapes; a 51-sample
  stress check of the long-`K` shape measured 1.057-1.072x within that band, while `M=129` fell to
  0.861x;
- sampled `225 <= M <= 256` points cleared 1.06x on every weight shape, while the adjacent `M=224`
  and `M=257` rows regressed to 0.966x and 0.967x respectively;
- the captured `M=72` and `M=249` rows passed a 10-warmup/51-sample direct-launch stress run across
  every weight shape, with worst-shape speedups of 1.027x and 1.059x respectively.

These results support a tile-aligned, dimension-aware configuration policy, not one large `M` range.
Before changing production dispatch, rerun the complete counted workload with the proposed expanded
router and then validate it end to end. The raw screening receipts are in
`capture/prefill_bm32_sweep.json`, `capture/prefill_bm32_64_128_sweep.json`,
`capture/prefill_bm32_224_257_sweep.json`, `capture/prefill_bm32_k8704_stress.json`, and
`capture/prefill_bm32_captured_stress.json`.

## Why GEAK did not export round 4

Round 4 was the best measured round candidate but missed GEAK's default relative promotion margin:

```text
round-2 cumulative weighted speedup = 1.097036x
round-4 verified weighted speedup   = 1.110781x
relative improvement                = 1.253%
required promotion improvement      = 2.000%
```

`IMPROVED=false` therefore meant “below the configured promotion margin,” not “slower” or “incorrect.”
The fresh audit confirms that the round-4 specialization is a real improvement worth carrying into the
vLLM integration.

## Reporting issue found during revalidation

The old harness's displayed weighted result can mix frozen latency weights with fresh A/B speedups:

```text
sum(frozen_weight) / sum(frozen_weight / fresh_speedup)
```

When the live baseline drifts from the frozen receipt, that hybrid is neither the fresh counted-time
ratio nor the frozen-denominator ratio. It produced an apparent 1.065x during this audit even though the
two valid calculations were 1.103x fresh-to-fresh and 1.108x frozen-to-fresh. Upstream-facing reports
should calculate one of the two explicit ratios and identify its denominator.

## Remaining vLLM integration checks

Before merging upstream:

1. Move the rocWMMA source into vLLM's supported extension/build mechanism and remove runtime
   `load_inline` compilation.
2. Gate the native path on gfx1201, wave32, FP8 E4M3 inputs, BF16 output, block size 128x128, supported
   strides, and the validated decode shapes; retain the incumbent fallback for every other contract.
3. Add unit coverage for all captured shapes, padded B strides, randomized scales, output independence,
   and non-default streams.
4. Regenerate primed timing receipts with the corrected weighted aggregation.
5. Run the same offline request suite end to end and report prefill latency, inter-token latency,
   throughput, and total request latency separately.
6. Keep the clean FlyDSL round-4 implementation as follow-up work; reevaluate it after direct model-load
   preshuffling or an equivalent deployment-valid weight-layout solution removes the cache-miss penalty.
