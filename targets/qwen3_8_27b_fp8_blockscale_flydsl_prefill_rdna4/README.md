# Qwen3.8-27B FP8 block-scale — FlyDSL broad-prefill expansion

This target runs a 32-direction GEAK campaign to expand the current standalone
FlyDSL gfx1201 kernel beyond `M=64`. The frozen seed uses the existing optimized
FlyDSL routes for `M=1..64` and vLLM's live Triton kernel as the correct
denominator for larger prefill shapes. Candidate work adds local FlyDSL kernel
variants and coarse routing without modifying either source checkout.

## Coverage

| Surface | Cases | Weight | Purpose |
|---|---:|---:|---|
| Captured Qwen prefill | 35 | production `count × latency` | Primary ratio-of-sums score |
| Captured decode | 10 | 0 | Preserve M1/M2 FlyDSL behavior |
| Existing FlyDSL micro-routes | 9 | 0 | Guard every M1–M64 route family |
| Qwen boundary/generalization | 35 | 0 | Cover M=65..1024 and tile discontinuities |
| Non-Qwen portability | 20 | 0 | Prevent exact production-shape routing |
| K%128 decode | 8 | 0 | Guard uneven split-K and K=2176 |

The 117 cases cover 39 distinct M values, all five captured Qwen `(N,K)`
families, portable `(128,128)` and `(1024,2176)` families, and arbitrary padded
B row strides. Zero-weight cases remain correctness and per-case performance
gates.

The current broad-prefill target is intentionally raw-weight: preshuffling or a
persistent model-weight copy is not allowed. Its purpose is to determine how
far explicit FlyDSL WMMA/LDS/register control can outperform the current hybrid
without changing vLLM's storage contract.

Every optimizer must read
[`baseline/EXPERIMENT_CONTEXT.md`](baseline/EXPERIMENT_CONTEXT.md). It contains
the exact math/deployment contract, seed hashes, current micro-routing, prior
profiling result, proposed search groups, and promotion gates.

## Prepare or inspect the manifest

```bash
cd /app/GEAK
python targets/qwen3_8_27b_fp8_blockscale_flydsl_prefill_rdna4/build_manifest.py
python targets/qwen3_8_27b_fp8_blockscale_flydsl_prefill_rdna4/baseline/test_kernel.py --list
```

## Safe preflight

This verifies the pinned FlyDSL sources, gfx1201 targeting, manifest invariants,
and representative M1/M64/M65 correctness without starting GEAK:

```bash
cd /app/GEAK
GEAK_PREFLIGHT_ONLY=1 \
  bash targets/qwen3_8_27b_fp8_blockscale_flydsl_prefill_rdna4/run_geak.sh
```

## Start the 32-direction job

```bash
cd /app/GEAK
bash targets/qwen3_8_27b_fp8_blockscale_flydsl_prefill_rdna4/run_geak.sh \
  |& tee targets/qwen3_8_27b_fp8_blockscale_flydsl_prefill_rdna4/capture/geak.log
```

The job is pinned to GPU 0, uses `mode=optimize` and `target_language=flydsl`,
has a 32-direction budget, and will not apply results to the original source.
Its final patch remains in the GEAK experiment directory for review.

If the external FlyDSL seed changes, the launcher fails its SHA-256 fence. Review
the new seed and update the hashes and campaign memory deliberately; use
`ALLOW_FLYDSL_SEED_DRIFT=1` only for an explicitly non-comparable exploratory
run. Likewise, `ALLOW_OTHER_RDNA4=1` creates a portability run whose timing must
not be compared directly with the gfx1201 receipts.
