# Qwen3.8 FP8 M256 native HIP on RDNA4

Focused GEAK campaign to beat vLLM's Triton FP8 block-scale GEMM at
`M=256, N=8192, K=5120` on gfx1201 using a single native HIP kernel.

- Optimization budget: 16 independent directions.
- Score: one primary M256 case against the frozen Triton denominator.
- Hard gates: six nearby M/stride/N/K cases, all requiring native HIP coverage.
- Correctness: five random draws, FP32-per-K128 scale semantics, fresh-output and mutation checks.
- Prohibited: FlyDSL, Triton candidate code, AITER, CK, library GEMMs, exact-signature routing,
  preshuffled/persistent weights, memoization, and extra device dispatches.

Generate or refresh the manifests:

```bash
/opt/python/bin/python build_manifest.py
```

Validate without starting GEAK:

```bash
GEAK_PREFLIGHT_ONLY=1 bash run_geak.sh
```

Run the 16-direction campaign:

```bash
bash run_geak.sh
```

The baseline implementation is intentionally Triton because it is the immutable oracle. Accepted
candidate work must replace the timed route with native HIP for all seven cases. See
`baseline/EXPERIMENT_CONTEXT.md` for the full evidence and promotion contract.
