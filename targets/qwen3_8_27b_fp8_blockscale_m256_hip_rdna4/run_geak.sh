#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"
VLLM_SRC="${VLLM_SRC:-/app/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/opt/python/bin/python}"
CODEX_BIN="${CODEX_BIN:-}"

if [ -z "$CODEX_BIN" ]; then
    for candidate in /root/.vscode-server/extensions/openai.chatgpt-*-linux-x64/bin/linux-x86_64/codex; do
        [ -x "$candidate" ] && CODEX_BIN="$candidate"
    done
fi

[ -x "$PYTHON_BIN" ] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[ -x "$CODEX_BIN" ] || { echo "Missing Codex CLI; set CODEX_BIN explicitly." >&2; exit 2; }
[ -s "$ARGS" ] || { echo "Missing GEAK args: $ARGS" >&2; exit 2; }
[ -s "$TARGET_DIR/baseline/cases.json" ] || {
    echo "Missing cases.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -s "$TARGET_DIR/baseline/workload.json" ] || {
    echo "Missing workload.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -d "$VLLM_SRC" ] || { echo "Missing vLLM source: $VLLM_SRC" >&2; exit 2; }

export PATH="$(dirname "$PYTHON_BIN"):$(dirname "$CODEX_BIN"):$PATH"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_ALLOWED="0"

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX ($GEAK_GPU_ARCH_CLASS)." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "This campaign is calibrated for gfx1201. Set ALLOW_OTHER_RDNA4=1 to override." >&2
    exit 2
fi

TARGET_DIR="$TARGET_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import torch

target = Path(os.environ["TARGET_DIR"])
cases = json.loads((target / "baseline/cases.json").read_text())["cases"]
roles = Counter(case["coverage_role"] for case in cases)
assert len(cases) == 7, f"expected 7 cases, got {len(cases)}"
assert roles["scored_m256_primary"] == 1
assert roles["zero_weight_generality_gate"] == 6
assert sum(case["weight"] > 0 for case in cases) == 1
assert all(case["N"] % 128 == 0 and case["K"] % 128 == 0 for case in cases)
assert all(case["b_stride"][1] == 1 and case["b_stride"][0] >= case["K"] for case in cases)
props = torch.cuda.get_device_properties(0)
print(
    f"M256 native-HIP preflight: {props.name}, gfx={os.environ['GEAK_GPU_GFX']}, "
    f"cases={len(cases)}, scored=1, physical_cus={os.environ['GEAK_GPU_CU_COUNT']}, "
    f"wgps={os.environ['GEAK_GPU_WGP_COUNT']}"
)
PY

if [ "${GEAK_PREFLIGHT_ONLY:-0}" = "1" ]; then
    "$PYTHON_BIN" "$TARGET_DIR/baseline/test_kernel.py" \
        --case m256_hip_primary_m256_n8192_k5120_bn128_bk128 \
        --case m256_hip_m257_tail_padded_m257_n8192_k5120_bn128_bk128 \
        --case m256_hip_m256_n5120_k8704_padded_m256_n5120_k8704_bn128_bk128 \
        --draws 1 --correctness-only
    echo "GEAK preflight-only validation passed; the 16-direction job was not started."
    exit 0
fi

mkdir -p "$TARGET_DIR/capture"
cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
