#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"

[ -s "$TARGET_DIR/baseline/cases.json" ] || {
    echo "Missing cases.json; run $TARGET_DIR/build_manifest.py first." >&2
    exit 2
}
[ -s "$TARGET_DIR/baseline/workload.json" ] || {
    echo "Missing workload.json; run $TARGET_DIR/build_manifest.py first." >&2
    exit 2
}
[ -s "$ARGS" ] || { echo "Missing GEAK args: $ARGS" >&2; exit 2; }

export VLLM_SRC="${VLLM_SRC:-/app/vllm}"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
# This target owns GPU 0 only. The allocation fence makes any accidental GPU 1
# request fail instead of consuming the card reserved for another bakeoff.
export GEAK_GPU_ALLOWED="0"

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX ($GEAK_GPU_ARCH_CLASS)." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "Receipts are gfx1201-specific; detected $GEAK_GPU_GFX. Set ALLOW_OTHER_RDNA4=1 for a new portability run." >&2
    exit 2
fi

python3 - <<'PY'
from pathlib import Path
import torch

vllm_src = Path(__import__("os").environ["VLLM_SRC"])
required = vllm_src / "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
assert required.is_file(), f"missing vLLM FP8 baseline: {required}"
props = torch.cuda.get_device_properties(0)
print(
    f"RDNA4 prefill target (GPU 0 only): {props.name}, {props.gcnArchName}, "
    f"physical_cus={__import__('os').environ['GEAK_GPU_CU_COUNT']}, "
    f"wgps={__import__('os').environ['GEAK_GPU_WGP_COUNT']}"
)
PY

mkdir -p "$TARGET_DIR/capture"
cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
