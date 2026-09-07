# Qwen3.8-27B FP8 block-scale — RDNA4 prefill specialization

This is the second-stage GEAK target for the Qwen3.8-27B-FP8 block-scaled linear operator on
`gfx1201`. It starts from the already validated Triton + HIP/rocWMMA hybrid, freezes that hybrid as
the new denominator, and searches only for additional prefill improvements. Decode is a protected
regression surface, not part of this target's score.

The run is intentionally a one-language bakeoff (`backends=["triton"]`). That preserves the
bakeoff freezer's immutable live-baseline oracle while dispatching only the incumbent Triton lane;
it does not start FlyDSL, AITER, CK, or a fresh authoring lane.

## What is scored

| Case class | Cases | Score weight | Purpose |
|---|---:|---:|---|
| Captured Qwen prefill | 35 | `count × validated seed latency` | Primary production metric |
| Captured decode | 10 | 0 | Preserve the shipped rocWMMA decode win |
| Synthetic K%128 decode sentinels | 8 | 0 | Protect odd split-K coverage, including K=2176 |
| Synthetic prefill sentinels | 30 | 0 | Correctness and bucket/generalization regression gate |

The primary number is the counted prefill ratio of sums. The checked-in seed weights total
approximately 735.78 ms over the captured offline workload. Synthetic sentinels never pretend to
be production frequency data. They cover the observed tile discontinuities, two full five-shape
cross-sections at `M=96` and `M=240`, and unseen interior buckets at M=192/384/640/832.

Every engineer must read [`baseline/EXPERIMENT_CONTEXT.md`](baseline/EXPERIMENT_CONTEXT.md). It
records the successful routes, the failed configurations, the BM32 crossover receipts, and the
acceptance gates. This is how the new run avoids rediscovering the first campaign.

## Run

```bash
cd /app/GEAK
bash targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/run_geak.sh \
  |& tee targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/capture/geak.log
```

The checked-in defaults pin this bakeoff to GPU 0 only, leaving GPU 1 free for a separate bakeoff.
An allocation fence rejects accidental GPU 1 use. The run has a 12-direction budget, a 0.5%
cumulative promotion gate, and three non-improving rounds before stopping. Override the whole
argument file with `GEAK_ARGS=/absolute/path/to/args.json`; do not edit the generated experiment
while it is running. An override must also remain on GPU 0 because the runner enforces that fence.

The runner refuses a non-`gfx1201` device by default because the baseline, profile, and crossover
receipts were all collected on an R9700. Set `ALLOW_OTHER_RDNA4=1` only for a new portability
campaign; its numbers must not be compared directly with this target's receipts.

## Refresh the manifest

`build_manifest.py` reconstructs `baseline/cases.json` and `baseline/workload.json` from the original
capture plus round-4 engineer 1's validated receipt:

```bash
python3 targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/build_manifest.py
```

The generated JSON is checked in, so the old timestamped experiment directory is provenance rather
than a runtime dependency. The seed source itself is also vendored under `baseline/`.

## Acceptance after GEAK

A candidate is integration-worthy only when all of these hold in fresh, paired measurements:

1. all 83 cases pass five randomized live-baseline parity draws and output-independence checks;
2. the counted prefill ratio-of-sums improves by at least 0.5% over the cumulative seed;
3. every captured and synthetic K%128 decode case remains at least 0.98x and `binding.cpp` retains
   its checked-in SHA-256;
4. every synthetic sentinel remains at least 0.98x, with no broad route inferred from one exact M;
5. the final result repeats in three full baseline/candidate A/B passes with reversed order and
   cache-flushed CUDA-event timing; and
6. no activation/output memoization, persistent result buffers, weight preshuffle, extra dispatch,
   or relaxed FP32-per-128-K scaling contract is introduced.

Small prefill gains are expected. A 1–4% complete-operator improvement is useful here; a spectacular
number is more likely a weighting, routing, or memoization error and must be audited before promotion.

## Deterministic simplified-route bakeoff

The upstream-oriented selector bypasses every candidate's existing hardcoded router and directly
compares the vLLM fallback, BM32, BM64 fused-scale, and BM80 shared-B kernels. Its default grid covers
33 M values across all five captured Qwen N/K weight families: every captured M, every routing
boundary, unseen bucket interiors, and both sides of the M=784 specialization. That produces 165
shapes in the definitive receipt.

Run the definitive selection with:

```bash
cd /app/GEAK
bash targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/run_prefill_selection.sh \
  |& tee targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/capture/prefill_selection.log
```

Use `PREFILL_QUICK=1` for a one-pass screening run. A definitive run uses three rotated/reversed-order
passes, 10 warmups, 51 cold-cache samples per leg, three correctness draws, and the GEAK GPU lock. Set
`PREFILL_RESUME=1` to continue an interrupted receipt without repeating completed shapes.
This selection runner is likewise fixed to GPU 0 and rejects any other `PREFILL_GPU_ID`.

The selector itself has a GPU-free regression test for preference ties, noisy measurements,
under-sampled buckets, per-shape regressions, generated-route behavior, and safe fallback:

```bash
python3 -m unittest \
  targets/qwen3_8_27b_fp8_blockscale_prefill_rdna4/test_prefill_route_selector.py
```

[`prefill_selection_policy.json`](prefill_selection_policy.json) keeps routing deliberately coarse:
six M buckets, three N buckets, and two K buckets. A custom variant owns a bucket only when every
sampled shape is correct and output-independent, the bucket reaches 1.01x, every paired pass stays at
least 0.98x, and timing spread is within 8%. At most two custom variants may survive. Policies within
0.5% choose fewer variants; per-bucket candidates within 1% prefer BM64, then BM32, then BM80.

The selector writes:

- `capture/prefill_variant_bench.json`: immutable raw per-pass timings and correctness;
- `capture/prefill_routes.json`: gates, competing policies, and the selected coarse routes;
- `capture/generated_prefill_routes.py`: explicit `select_prefill_variant(M, N, K)` implementation;
- `capture/PREFILL_ROUTE_SELECTION.md`: concise human review table.

Unlisted or under-sampled buckets always return `baseline`. Captured production weights are reported
as a diagnostic, but policy selection uses equal case weights so a Qwen-specific hot shape cannot buy
an unsafe generalization rule.
