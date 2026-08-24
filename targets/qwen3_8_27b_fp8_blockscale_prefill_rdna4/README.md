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

The checked-in defaults use GPUs 0 and 1, a 12-direction budget, a 0.5% cumulative promotion gate,
and three non-improving rounds before stopping. Override the whole argument file with
`GEAK_ARGS=/absolute/path/to/args.json`; do not edit the generated experiment while it is running.

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
