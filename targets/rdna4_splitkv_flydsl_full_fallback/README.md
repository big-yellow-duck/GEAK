# RDNA4 FlyDSL full SplitKV fallback expansion

This is a standalone GEAK target for replacing vLLM's remaining RDNA4 Triton
SplitKV paged-decode domain with FlyDSL. The frozen hybrid seed uses the current
validated FlyDSL GQA-6/7 winner and live Triton everywhere else.

## Coverage

The 66-case manifest contains 58 scored fallback cases and eight hard guards.
It covers GQA 1-16, head dimensions 128/256, BF16/FP16 queries and matching
caches, FP8 E4M3FN/E4M3FNUZ caches with non-unit scales, page sizes 16-1568,
multiple KV-head counts, arbitrary scale, batch one, and ragged B3/B8.

Generate or inspect the manifest:

```bash
cd /app/GEAK
/opt/python/bin/python \
  targets/rdna4_splitkv_flydsl_full_fallback/build_manifest.py
/opt/python/bin/python \
  targets/rdna4_splitkv_flydsl_full_fallback/baseline/test_kernel.py --list
```

Run the representative GPU preflight on physical GPU 1:

```bash
cd /app/GEAK
GEAK_PREFLIGHT_ONLY=1 \
  bash targets/rdna4_splitkv_flydsl_full_fallback/run_geak.sh
```

Start the 32-direction campaign:

```bash
cd /app/GEAK
bash targets/rdna4_splitkv_flydsl_full_fallback/run_geak.sh \
  |& tee targets/rdna4_splitkv_flydsl_full_fallback/capture/geak.log
```

The run is pinned to physical GPU 1, uses `mode=optimize` and
`target_language=flydsl`, and never applies a result to the external FlyDSL or
vLLM checkout. Candidate patches and reports remain under `/app/GEAK/exp/`.

The launcher SHA-fences both external seed files and the Triton reference. If
either changes, review and deliberately update the campaign contract; do not
silently compare results from different denominators.
