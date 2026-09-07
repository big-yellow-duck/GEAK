#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"
PYTHON_BIN="${GEAK_PYTHON:-/opt/python/bin/python}"

for required in \
    "$TARGET_DIR/baseline/kernel.py" \
    "$TARGET_DIR/baseline/test_kernel.py" \
    "$TARGET_DIR/baseline/cases.json" \
    "$TARGET_DIR/baseline/workload.json" \
    "$TARGET_DIR/baseline/EXPERIMENT_CONTEXT.md" \
    "$ARGS"; do
    [ -s "$required" ] || { echo "Missing GEAK input: $required" >&2; exit 2; }
done
[ -x "$PYTHON_BIN" ] || { echo "Missing Python environment: $PYTHON_BIN" >&2; exit 2; }

export VLLM_SRC="${VLLM_SRC:-/app/vllm-rdna4-fp8-flydsl-tp2-hip-ar}"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"

# Keep GEAK's allocation in host-physical coordinates. gpu_lock.sh owns the
# per-command HIP_VISIBLE_DEVICES=1 selection, its physical GPU 1 flock, and its
# physical GPU 1 idle check. Clear inherited selectors so they cannot intersect.
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export GEAK_GPU_ALLOWED="1"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_USE_LOG="$TARGET_DIR/capture/gpu_use.jsonl"

eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX ($GEAK_GPU_ARCH_CLASS)." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "This campaign is gfx1201-specific; detected $GEAK_GPU_GFX." >&2
    exit 2
fi

mkdir -p "$TARGET_DIR/capture"
bash "$GEAK_ROOT/kernel_workflow/scripts/gpu_lock.sh" 1 "$PYTHON_BIN" - <<'PY'
import os
import torch

assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
assert torch.cuda.device_count() == 1, "GPU visibility fence did not expose exactly one GPU"
props = torch.cuda.get_device_properties(0)
assert props.gcnArchName == "gfx1201", props.gcnArchName
print(
    "GEAK FP8 SplitKV: host GPU id 1 (logical device 0 inside benchmark); "
    f"{props.name}, {props.gcnArchName}, uuid={props.uuid}; budget=16"
)
PY

if [ "${GEAK_PREFLIGHT_ONLY:-0}" = "1" ]; then
    "$PYTHON_BIN" "$TARGET_DIR/baseline/test_kernel.py" \
        --case guard_b1_s4014 --case split_b1_s8192 \
        --case guard_bf16_b1_s1569 --case guard_bf16_ragged_b3_s8192 \
        --check --warmup 2 --repeats 5
    echo "GEAK FP8/BF16 oracle preflight passed; workflow not started."
    exit 0
fi

cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
