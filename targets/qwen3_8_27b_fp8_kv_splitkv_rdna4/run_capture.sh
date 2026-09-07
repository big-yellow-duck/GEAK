#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
VLLM_SRC="${VLLM_SRC:-/app/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/opt/python/bin/python}"
RAW="$TARGET_DIR/capture/raw_calls.jsonl"
MANIFEST="$TARGET_DIR/capture/manifest.json"
LOG="$TARGET_DIR/capture/capture.log"

[ -x "$PYTHON_BIN" ] || {
    echo "Missing vLLM environment Python: $PYTHON_BIN" >&2
    exit 2
}
[ -d "$VLLM_SRC/vllm" ] || { echo "Missing vLLM source: $VLLM_SRC" >&2; exit 2; }

mkdir -p "$TARGET_DIR/capture"
: > "$RAW"
: > "$LOG"

export PYTHONPATH="$TARGET_DIR/probe:$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES="${CAPTURE_GPUS:-0,1}"
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

eval "$(GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This exploration target requires RDNA4; detected $GEAK_GPU_GFX." >&2
    exit 2
fi

"$PYTHON_BIN" "$TARGET_DIR/capture_shapes.py" \
    --model "${MODEL:-Qwen/Qwen3.8-27B-FP8}" \
    --revision "${REVISION:-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}" \
    --vllm-src "$VLLM_SRC" \
    --tp "${TP:-2}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --shape-log "$RAW" \
    --manifest "$MANIFEST" |& tee "$LOG"

echo "Capture ready:  $RAW"
echo "Manifest ready: $MANIFEST"
