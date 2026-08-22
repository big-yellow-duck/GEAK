---
myst:
    html_meta:
        "description": "Install GEAK 4.0.0 with Codex CLI or Claude Code, ROCm, and a serving backend."
        "keywords": "GEAK, install, ROCm, Codex CLI, Claude Code, Workflow, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK

GEAK 4.0.0 consists of deterministic workflows (`e2e_workflow.js` / `kernel_workflow.js`)
that run through Codex CLI or Claude Code. "Installing" means: get the repo, get an agent harness, and have a
working ROCm environment (plus a serving backend for E2E). For a first run, see
[Run a workflow](../how-to/run-agent.md).

## Prerequisites

GEAK 4.0.0 requires the following software and hardware.

| Requirement | Detail |
|---|---|
| **AMD Instinct™ MI GPU** | CDNA, gfx942 (MI300X) / gfx950 (MI350X/MI355X). Auto-detected. |
| **ROCm 6+** | `rocminfo` / `rocm-smi` must work. |
| **A profiler** | One of `rocprof-compute`, `rocprofv3`, `rocprof` (also `omniperf` or `metrix`). Auto-detected. |
| **Python 3.8+** | Tested on 3.12. |
| **Codex CLI + Node.js 18+ (default)** | Run `codex login`; ChatGPT subscription authentication is supported. Node evaluates the portable JS workflow runtime. |
| **Claude Code ≥ 2.1.177 (optional)** | Legacy backend selected with `GEAK_AGENT_BACKEND=claude`. |
| **Serving backend (E2E)** | A running-capable `sglang` or `vllm`, plus model weights on disk. |

## Set up GEAK

Clone the repository and run the setup script.

Installing GEAK installs the `geak` Python package + deps, clones the repo, and installs the selected agent CLI.
By default the repo lands in `./GEAK` under the directory you run the command from (override with `GEAK_HOME`).
Pick either method — both end up the same:

**A. One-liner** — run it in the directory where you want GEAK to live:

```bash
pip install "git+https://github.com/AMD-AGI/GEAK"
```

**B. Clone first** — if you'd rather have the checkout up front (e.g. to work on a branch):

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK
pip install .
```

Codex is the default. Authenticate once using the ChatGPT account already associated with your subscription:

```bash
codex login
codex login status
```

Launch GEAK:

```bash
GEAK_AGENT_BACKEND=codex python interface/run_e2e.py handoff.json result.json
```

For Claude compatibility, install with `GEAK_AGENT_BACKEND=claude`, configure its API key/gateway or login,
and launch as before. Nothing is compiled at clone time; workflow sources and their role/knowledge files are
used directly.

## Verify the environment

Run these checks before starting a workflow. A misconfigured environment fails deep into a multi-hour run.

```bash
# Codex and subscription authentication
codex --version
codex login status

# Portable workflow evaluator
node --version

# GPU is visible to ROCm
rocminfo | grep -E "Name:|gfx"

# At least one profiler is on PATH
command -v rocprof-compute || command -v rocprofv3 || command -v rocprof
```

Expected output:

- `codex login status` reports a logged-in ChatGPT session and Node.js is 18 or newer.
- `rocminfo` lists your GPU name and a `gfx942` or `gfx950` target.
- At least one profiler command resolves without error.

If `rocminfo` fails, your ROCm stack is not installed or not on PATH. If no profiler resolves, install `rocprof-compute` (preferred) or `rocprofv3`.

## Related topics

- [Run a workflow](../how-to/run-agent.md): start a single-kernel or end-to-end run.
- [Compatibility matrix](../compatibility.md): verified GPUs, ROCm versions, backends, and dtypes.
