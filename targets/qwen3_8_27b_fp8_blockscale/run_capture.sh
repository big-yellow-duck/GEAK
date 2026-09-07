#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
VLLM_SRC="${VLLM_SRC:-/app/vllm}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
REVISION="${REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
RAW="$TARGET_DIR/capture/raw_calls.jsonl"
MANIFEST="$TARGET_DIR/capture/manifest.json"
TIMINGS="$TARGET_DIR/capture/baseline_timings.json"
ARGS="$TARGET_DIR/bakeoff.args.json"

mkdir -p "$TARGET_DIR/capture"
if [ "${RESET_CAPTURE:-1}" = 1 ]; then
    : > "$RAW"
fi

PROBE_SOURCE="$VLLM_SRC/vllm/model_executor/kernels/linear/scaled_mm/triton.py"
if ! rg -q 'geak\.w8a8_blockscale_call\.v1' "$PROBE_SOURCE"; then
    echo "Installing the opt-in GEAK shape probe into $VLLM_SRC"
    git -C "$VLLM_SRC" apply "$TARGET_DIR/vllm_shape_logging.patch"
fi

export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_PLUGINS="${VLLM_PLUGINS-}"
export HIP_VISIBLE_DEVICES="${CAPTURE_GPUS:-0,1}"

eval "$(GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-$GEAK_GPU_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"

python3 "$TARGET_DIR/capture_shapes.py" \
    --model "$MODEL" --revision "$REVISION" --tp "${TP:-2}" \
    --max-model-len "${MAX_MODEL_LEN:-2048}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --shape-log "$RAW" --manifest "$MANIFEST"

python3 "$TARGET_DIR/aggregate_shapes.py" \
    --shape-log "$RAW" --manifest "$MANIFEST" \
    --baseline-dir "$TARGET_DIR/baseline" --bakeoff-json "$ARGS" \
    --gpu-ids "${GEAK_GPU_IDS:-0,1}" --budget "${GEAK_BUDGET:-8}"

HIP_VISIBLE_DEVICES="${BASELINE_GPU:-0}" VLLM_SRC="$VLLM_SRC" \
    python3 "$TARGET_DIR/baseline/test_kernel.py" \
      --warmup "${BASELINE_WARMUP:-5}" --repeats "${BASELINE_REPEATS:-20}" \
      --benchmark-json "$TIMINGS"

python3 "$TARGET_DIR/aggregate_shapes.py" \
    --shape-log "$RAW" --manifest "$MANIFEST" --timings "$TIMINGS" \
    --baseline-dir "$TARGET_DIR/baseline" --bakeoff-json "$ARGS" \
    --gpu-ids "${GEAK_GPU_IDS:-0,1}" --budget "${GEAK_BUDGET:-8}"

echo "Capture ready: $MANIFEST"
echo "Cases ready:   $TARGET_DIR/baseline/cases.json"
echo "Bakeoff args:  $ARGS"
