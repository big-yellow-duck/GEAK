---
myst:
    html_meta:
        "description": "Verified hardware, software, agent runtime, and backend combinations for GEAK 4.0.0."
        "keywords": "GEAK, compatibility, ROCm, AMD Instinct, Codex CLI, Claude Code, vLLM"
---

# GEAK compatibility matrix

Verified hardware, software, runtime, and backend combinations for GEAK 4.0.0, a deterministic
JS-workflow GPU optimizer supporting Codex CLI and Claude Code.

Use the following matrix to view the compatibility and system requirements:

| AMD Instinct GPU | ROCm version | Python | Ubuntu |
|---|---|---|---|
| MI300X, MI325X, MI355X| 6.4x, 7.0x, 7.1x, 7.2x | 3.8, 3.12 | 22.04, 24.04 | 

```{note}
- The on-box card is auto-detected (`rocminfo` / `rocm_agent_enumerator`); `PYTORCH_ROCM_ARCH` is pinned
to the local `gfx` at build time.
- The on-box GPU arch, backend, profiler, and op backends are re-checked at runtime by the Setup
  preflight, which writes `env_report.{md,json}`.
- Only tested configurations are listed. To report a verified configuration not listed here, open a pull request.
- For the Python version, the compiled artifacts in the tree are cpython-312.
```

## Agent runtime

| Component | Version | Notes |
|---|---|---|
| Codex CLI | Current CLI plus Node.js 18+ | Default local backend; authenticates with `codex login` and can use ChatGPT subscription access. |
| Codex model | `gpt-5.6-sol` / `high` | Configurable with `GEAK_CODEX_MODEL` and `GEAK_CODEX_REASONING_EFFORT`. |
| Claude Code | ≥ 2.1.177 | Optional legacy backend selected with `GEAK_AGENT_BACKEND=claude`. |
| Permissions | Controlled host/container only | Agents run profiling, builds, and benchmarks with full filesystem/process access. |

## Invocation mode

| Mode | Notes |
|---|---|
| Codex compatibility runtime | `node interface/codex_workflow_runner.mjs --script ... --args-file ...` |
| Claude natural language → `Workflow` | Legacy Claude Code discovery/invocation path. |
| Direct `Workflow` call (e2e) | `scriptPath: "<repo>/e2e_workflow/e2e_workflow.js"` | 
| Direct `Workflow` call (single kernel) | `scriptPath: "<repo>/kernel_workflow/kernel_workflow.js"` | 
| External orchestrator (Hyperloom) | `python interface/run_e2e.py <handoff.json> <result.json>` | 

## Profilers

The profiler is auto-detected using a degrade ladder
(`PROFILER_PRIORITY="rocprof-compute rocprofv3 rocprof metrix"`).

- `rocprof-compute`
- `rocprofv3` 
- `rocprof`
- `metrix`

## Serving backends (e2e_workflow)

The serving stack is not baked in; `args.backend` selects `scripts/adapters/<backend>.sh`.

| Backend | Default port | 
|---|---|
| sglang | 30000 | 
| vllm | 8000 | 

| Bench client | Notes |
|---|---|
| Backend-native (`bench_e2e.sh`) | Default dispatcher. |
| inferencex | Opt-in using `BENCH_CLIENT=inferencex` (Hyperloom or Magpie parity); needs `$INFERENCEX_PATH`. |

## Kernel languages (kernel_workflow)

`target_language` for author mode; also the languages the head-kernel bake-off can win with.

| Kernel language | Notes |
|---|---|
| Triton | Always a viable author target. |
| FlyDSL | Preferred author target for dense / quantized GEMM (aiter's SOTA GEMM DSL, JIT, no build). Probed using `aiter.ops.flydsl.is_flydsl_available()`. |
| HIP | Used when headroom justifies it. |
| CK (Composable Kernel) | Used when headroom justifies it; FP8 GEMM tuning. |

## Op-backend bake-off (`op_bench.py` head kernels)

- hipblaslt 
- rocblas or TunableOp
- aiter 
- triton (with `--triton-autotune`) 

## Precision and data types

| Data type | Notes |
|---|---|
| FP16 / BF16 | General kernel optimization. |
| FP8 | Head-GEMM tuning + author target; gsm8k accuracy gate recommended for quantized kernels. |
| FP4 / MXFP | Quantized GEMM (A4W4); accuracy gate recommended. |

## Accuracy gate (e2e_workflow)

| Gate | Requirement | 
|---|---|
| `none` (default) | Throughput delta + greedy output parity. |
| `gsm8k` | Sampled gsm8k (5-shot, greedy, fixed seed) using `scripts/gsm8k_eval.py` against an OpenAI-compatible `/v1` endpoint. | 
