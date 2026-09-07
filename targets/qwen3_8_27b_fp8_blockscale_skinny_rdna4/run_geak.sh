#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"
export VLLM_SRC="${VLLM_SRC:-/app/vllm}"
PYTHON_BIN="${PYTHON_BIN:-$VLLM_SRC/.venv/bin/python}"
ROCM_PYTHON_SITE="${ROCM_PYTHON_SITE:-/opt/python/lib/python3.14/site-packages}"

[ -x "$PYTHON_BIN" ] || { echo "Missing vLLM Python: $PYTHON_BIN" >&2; exit 2; }
[ -d "$ROCM_PYTHON_SITE/torch" ] || {
    echo "Missing ROCm torch package under $ROCM_PYTHON_SITE" >&2
    exit 2
}
[ -s "$TARGET_DIR/baseline/cases.json" ] || {
    echo "Missing cases.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -s "$TARGET_DIR/baseline/workload.json" ] || {
    echo "Missing workload.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -s "$ARGS" ] || { echo "Missing GEAK args: $ARGS" >&2; exit 2; }

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$VLLM_SRC:$ROCM_PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_ALLOWED="0"

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "Receipts are gfx1201-specific. Set ALLOW_OTHER_RDNA4=1 for a portability run." >&2
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import torch

target = Path(os.environ.get(
    "SKINNY_TARGET_DIR",
    "/app/GEAK/targets/qwen3_8_27b_fp8_blockscale_skinny_rdna4",
))
cases = json.loads((target / "baseline/cases.json").read_text())["cases"]
ms = sorted({int(case["M"]) for case in cases if case["coverage_role"] == "qwen_skinny"})
assert ms == list(range(1, 17)), f"incomplete Qwen M coverage: {ms}"
assert len(cases) == 112, f"expected 112 cases, got {len(cases)}"
props = torch.cuda.get_device_properties(0)
print(f"RDNA4 native skinny target: {props.name}, {props.gcnArchName}, cases={len(cases)}")
PY

mkdir -p "$TARGET_DIR/capture"
cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
