#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${BAKEOFF_ARGS:-$TARGET_DIR/bakeoff.args.json}"

[ -s "$TARGET_DIR/baseline/cases.json" ] || {
    echo "Missing captured cases. Run $TARGET_DIR/run_capture.sh first." >&2
    exit 2
}
[ -s "$ARGS" ] || { echo "Missing bakeoff args: $ARGS" >&2; exit 2; }

export VLLM_SRC="${VLLM_SRC:-/app/vllm}"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-$GEAK_GPU_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"

python3 - <<'PY'
import os
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
arch = get_rocm_arch()
assert is_rdna_arch(arch), f"FlyDSL is not targeting RDNA: {arch}"
print(
    f"FlyDSL target preflight: {arch}, wave{os.environ['GEAK_GPU_WAVE_SIZE']}, "
    f"physical_cus={os.environ['GEAK_GPU_CU_COUNT']}, "
    f"wgps={os.environ['GEAK_GPU_WGP_COUNT']}"
)
PY

cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
