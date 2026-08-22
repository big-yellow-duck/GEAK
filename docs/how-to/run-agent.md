---
myst:
    html_meta:
        "description": "Run a GEAK v4 workflow with Codex CLI or Claude Code."
        "keywords": "GEAK, run workflow, serving throughput, single kernel, Codex CLI, Claude Code, sglang, vLLM"
---

# Run a GEAK workflow

GEAK v4 uses deterministic JS workflows with Codex CLI or Claude Code. The stable
`interface/run_e2e.py` entry point selects the harness and keeps its handoff/result contract unchanged.

## Prerequisites

Before running a workflow, ensure the following are in place.

- **AMD Instinct™ MI GPU**: CDNA (gfx942 / gfx950), auto-detected.
- **ROCm 6+** with `rocminfo` / `rocm-smi`, and a profiler (`rocprof-compute` / `rocprofv3` / `rocprof`).
- **Python 3.8+**.
- **Codex CLI plus Node.js 18+** (default), authenticated with `codex login`; or Claude Code ≥2.1.177.
- **For E2E:** a running-capable `sglang` or `vllm` and the model weights on disk.

## Get the repo and authenticate Codex

Clone GEAK and use the normal Codex ChatGPT login:

```bash
git clone https://github.com/AMD-AGI/GEAK.git && cd GEAK
codex login
codex login status
```

The GEAK compatibility runner invokes `codex exec` non-interactively with schema-constrained results and
full permissions; use it only on a controlled GPU host/container.

## Run an end-to-end workflow

Use the same handoff/result interface used by external orchestrators:

```bash
GEAK_AGENT_BACKEND=codex \
GEAK_CODEX_MODEL=gpt-5.6-sol \
python interface/run_e2e.py /absolute/path/handoff.json /absolute/path/result.json
```

Profiles a running server, triages hot kernels by Amdahl (`pct_gpu_time × achievable_speedup`), pulls
levers cheapest-first (config/backend sweep → head GEMM/attention bake-off → editable-kernel milestone
loop), and overlays each accepted change back reversibly, gated on a measured throughput delta.

Output: `e2e_workflow/exp/e2e_<model>_<timestamp>/` — `final_report.md`, `architect_report.md`, `final/`
(overlay + patch + `final_launch.sh`).

## Run a single-kernel workflow

```json
{
  "kernel_path": "/absolute/path/to/kernel",
  "workflow_dir": "/absolute/path/to/GEAK/kernel_workflow",
  "mode": "optimize",
  "budget": 8,
  "gpu_ids": "0"
}
```

```bash
GEAK_CODEX_MODEL=gpt-5.6-sol node interface/codex_workflow_runner.mjs \
  --script kernel_workflow/kernel_workflow.js --args-file kernel-args.json
```

Director → TechLead → specialist engineers, multi-round and budget-controlled, each patch independently
verified. Output: `kernel_workflow/exp/team_<kernel>_<timestamp>/`.

**Batch:** run multiple kernels in parallel by launching one compatibility runner per
kernel. GPU access is serialized internally using `kernel_workflow/scripts/gpu_lock.sh`, so all
processes can safely share the same GPUs.

```bash
GEAK=/absolute/path/to/GEAK
GPU_IDS="0,1,2,3"

for KERNEL in /path/to/kernel_a /path/to/kernel_b /path/to/kernel_c; do
  ARGS="$KERNEL/geak-args.json"
  python -c 'import json,sys; json.dump({"kernel_path":sys.argv[1],
    "workflow_dir":sys.argv[3]+"/kernel_workflow","mode":"optimize",
    "budget":8,"gpu_ids":sys.argv[4]},open(sys.argv[2],"w"))' \
    "$KERNEL" "$ARGS" "$GEAK" "$GPU_IDS"
  node "$GEAK/interface/codex_workflow_runner.mjs" \
    --script "$GEAK/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS" > "$KERNEL/codex.log" 2>&1 &
done

wait
```

Each process writes its output to a per-kernel log. The `kernel_workflow` creates its experiment
directory under `kernel_workflow/exp/team_<kernel>_<timestamp>/`. Each process exits when its workflow completes.

## Depth modes (e2e)

Both default off = **default** mode; mutually exclusive, deep wins.

| Mode | Trigger | What runs |
|---|---|---|
| **default** | *(none)* | ConfigSweep + HeadKernel + Milestone. |
| **fast** | "fast mode" | HeadKernel only, parallel, time-boxed (`fast_budget_ms`, 5h). |
| **deep** | "deep mode" | ConfigSweep + HeadKernel, cross-kernel×backend lane pool, many rounds (`deep_head_budget_ms`, 24h). |

```
use /absolute/path/to/GEAK/e2e_workflow, deep mode, to optimize /models/Qwen3.5-27B-FP8 on gpus 0-7
```

## Accuracy gate (quantized kernels)

For FP8 / MXFP4, byte-parity is too strict — switch the e2e gate to task accuracy:

```
... use the gsm8k accuracy gate with limit 200 and tolerance 0.01
```

The Integrator then runs sampled gsm8k (5-shot, greedy, fixed seed) and accepts iff
`cand_em >= baseline_em - tol`.

## Related topics

- [Install GEAK](../install/install.md) 
- [API reference](../reference/api-reference.md) 
- [Compatibility matrix](../compatibility.md)
