# Qwen3.8-27B FP8 block-scale — RDNA4 native skinny GEMM

This GEAK target expands vLLM's native gfx12x HIP/rocWMMA block-FP8 GEMM from
`M=1,2` to complete `M=1..16` coverage. It is deliberately not an exact `M=4`
specialization: every M value must use the native candidate, pass live parity,
and remain within the per-case performance floor.

The production score is weighted by the five steady `M=4` Qwen decode shapes
captured after enabling `NCCL_PROTO=Simple`. Existing `M=1,2` native routes,
all unobserved `M=3..16` values, and small/odd-K portability sentinels are hard
gates. Zero-weight gates cannot be traded away for an `M=4` win.

## Workload

- 80 Qwen cases: every `M=1..16` across all five production `(N,K)` families.
- 32 portability cases: every `M=1..16` at `(N,K)=(128,128)` and
  `(1024,2176)`.
- FP8 E4M3 A/B, FP32 1x128 and 128x128 scales, FP32 scaled-partial
  accumulation, BF16 output, and captured padded weight strides.
- One device dispatch, fresh output storage, no preshuffled or persistent
  weight copy, and no result memoization.

The current baseline calls the existing native HIP path for `M=1,2` and the
generic vLLM Triton fallback for `M=3..16`. The candidate must replace that
split with a native HIP implementation covering the full interval. A source or
profile engagement check is required because correctness alone cannot detect a
hidden Triton fallback.

## Run

```bash
cd /app/GEAK
/app/vllm/.venv/bin/python \
  targets/qwen3_8_27b_fp8_blockscale_skinny_rdna4/build_manifest.py
bash targets/qwen3_8_27b_fp8_blockscale_skinny_rdna4/run_geak.sh \
  |& tee targets/qwen3_8_27b_fp8_blockscale_skinny_rdna4/capture/geak.log
```

The checked-in manifest is deterministic; rebuilding it should produce no
diff. The default job is pinned to GPU 0 and `gfx1201`.

## Promotion gates

1. Every `M=1..16` must engage one native HIP dispatch for every supported
   layout; exact `M=4` or exact Qwen-shape routing is rejected.
2. All 112 cases pass five randomized comparisons at BF16
   `rtol=0.01, atol=0.01`, including output-independence and rank-pattern
   inputs that expose row-mixing mistakes.
3. `M=1,2` retain at least 0.98x of the incumbent native performance on every
   Qwen family.
4. Every expansion and portability case retains at least 0.98x of the live
   hybrid baseline. The production-weighted `M=4` ratio-of-sums must improve by
   at least 1.01x.
5. Definitive timing uses three cache-flushed, reversed-order paired passes,
   followed by CUDA-graph replay and a full Qwen TP serving A/B with
   `NCCL_PROTO=Simple` held constant.
