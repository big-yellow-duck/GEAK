#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
GPU_ID="${PREFILL_GPU_ID:-0}"
CANDIDATE="${PREFILL_CANDIDATE:-$TARGET_DIR/variants/round4_kernel.py}"
CASES="${PREFILL_CASES:-$TARGET_DIR/baseline/cases.json}"
POLICY="${PREFILL_POLICY:-$TARGET_DIR/prefill_selection_policy.json}"
OUTPUT="${PREFILL_BENCH_OUTPUT:-$TARGET_DIR/capture/prefill_variant_bench.json}"
ROUTES="${PREFILL_ROUTES_OUTPUT:-$TARGET_DIR/capture/prefill_routes.json}"
ROUTES_PY="${PREFILL_ROUTES_PYTHON:-$TARGET_DIR/capture/generated_prefill_routes.py}"
ROUTES_MD="${PREFILL_ROUTES_MARKDOWN:-$TARGET_DIR/capture/PREFILL_ROUTE_SELECTION.md}"

export VLLM_SRC="${VLLM_SRC:-/app/vllm}"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_ALLOWED="0"

if [ "$GPU_ID" != "0" ]; then
    echo "This target is reserved to GPU 0; requested PREFILL_GPU_ID=$GPU_ID." >&2
    exit 2
fi

if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "Selection receipts are gfx1201-specific; detected $GEAK_GPU_GFX." >&2
    echo "Set ALLOW_OTHER_RDNA4=1 only to produce a separate portability receipt." >&2
    exit 2
fi

PASSES="${PREFILL_PASSES:-3}"
WARMUP="${PREFILL_WARMUP:-10}"
REPEATS="${PREFILL_REPEATS:-51}"
DRAWS="${PREFILL_CORRECTNESS_DRAWS:-3}"
if [ "${PREFILL_QUICK:-0}" = "1" ]; then
    PASSES=1
    WARMUP=3
    REPEATS=11
    DRAWS=1
fi

mkdir -p "$TARGET_DIR/capture"
BENCH_ARGS=(
    python3 "$TARGET_DIR/benchmark_prefill_variants.py"
    --candidate "$CANDIDATE"
    --cases "$CASES"
    --output "$OUTPUT"
    --variants "${PREFILL_VARIANTS:-bm32,bm64,bm80}"
    --passes "$PASSES"
    --warmup "$WARMUP"
    --repeats "$REPEATS"
    --correctness-draws "$DRAWS"
    --cache-flush-mb "${PREFILL_CACHE_FLUSH_MB:-512}"
)
if [ -n "${PREFILL_M_VALUES:-}" ]; then
    BENCH_ARGS+=(--m-values "$PREFILL_M_VALUES")
fi
if [ "${PREFILL_RESUME:-0}" = "1" ]; then
    BENCH_ARGS+=(--resume)
fi

echo "Prefill selection benchmark: gfx=$GEAK_GPU_GFX gpu=$GPU_ID passes=$PASSES repeats=$REPEATS"
bash "$GEAK_ROOT/kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" "${BENCH_ARGS[@]}"

python3 "$TARGET_DIR/select_prefill_routes.py" \
    --bench "$OUTPUT" \
    --policy "$POLICY" \
    --output "$ROUTES" \
    --python-output "$ROUTES_PY" \
    --markdown-output "$ROUTES_MD"

echo "Route report: $ROUTES_MD"
