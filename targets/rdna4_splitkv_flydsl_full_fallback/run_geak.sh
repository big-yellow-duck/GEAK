#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"
PYTHON_BIN="${GEAK_PYTHON:-/opt/python/bin/python}"
export FLYDSL_SEED_ROOT="${FLYDSL_SEED_ROOT:-/app/FlyDSL}"
export VLLM_SRC="${VLLM_SRC:-/app/vllm-rdna4-fp8-flydsl-tp2-hip-ar}"

for required in \
    "$ARGS" \
    "$TARGET_DIR/baseline/kernel.py" \
    "$TARGET_DIR/baseline/test_kernel.py" \
    "$TARGET_DIR/baseline/cases.json" \
    "$TARGET_DIR/baseline/workload.json" \
    "$TARGET_DIR/baseline/EXPERIMENT_CONTEXT.md"; do
    [ -s "$required" ] || { echo "Missing GEAK input: $required" >&2; exit 2; }
done
[ -x "$PYTHON_BIN" ] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[ -d "$FLYDSL_SEED_ROOT/build-fly/python_packages/flydsl" ] || {
    echo "Missing built FlyDSL package under $FLYDSL_SEED_ROOT/build-fly" >&2
    exit 2
}

check_seed_sha() {
    local expected="$1"
    local path="$2"
    local label="$3"
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        if [ "${ALLOW_SPLITKV_SEED_DRIFT:-0}" = "1" ]; then
            echo "WARNING: $label drift: expected $expected, got $actual" >&2
        else
            echo "$label drift: expected $expected, got $actual" >&2
            echo "Review the new denominator or set ALLOW_SPLITKV_SEED_DRIFT=1." >&2
            exit 2
        fi
    fi
}

check_seed_sha \
    d6ae8de377ee97f048f10ad2b24abe6adc1b4c9db5214e06ccfc438d1661835e \
    "$FLYDSL_SEED_ROOT/kernels/attention/rdna4_splitkv/kernel.py" \
    "FlyDSL SplitKV launcher"
check_seed_sha \
    f97954e94dde81a29e6668bc0638a5d64d94c8b033acf3f315506d511623a2bb \
    "$FLYDSL_SEED_ROOT/kernels/attention/rdna4_splitkv/wmma.py" \
    "FlyDSL grouped WMMA stage"
check_seed_sha \
    47bf4f068250002e1275956841b7441f46ca89a7cb761178f0544e4d5fec0060 \
    "$FLYDSL_SEED_ROOT/kernels/common/tensor_shim.py" \
    "FlyDSL tensor launcher"
check_seed_sha \
    84bcaf3ba33f7a87f20bda1de267081b1b390225c5698ac2f8b4eca87cffcf31 \
    "$VLLM_SRC/vllm/v1/attention/ops/chunked_prefill_paged_decode.py" \
    "vLLM Triton SplitKV reference"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLYDSL_SEED_ROOT/build-fly/python_packages:$FLYDSL_SEED_ROOT:$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FLYDSL_SEED_ROOT/build-fly/lib:$FLYDSL_SEED_ROOT/build-fly/python_packages/flydsl/_mlir/_mlir_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-$GEAK_GPU_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_ALLOWED="1"
export GEAK_GPU_USE_LOG="$TARGET_DIR/capture/gpu_use.jsonl"
export FLYDSL_RUNTIME_CACHE_DIR="${FLYDSL_RUNTIME_CACHE_DIR:-$TARGET_DIR/capture/flydsl_cache}"

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "Receipts are gfx1201-specific; detected $GEAK_GPU_GFX." >&2
    exit 2
fi

mkdir -p "$TARGET_DIR/capture" "$FLYDSL_RUNTIME_CACHE_DIR"
TARGET_DIR="$TARGET_DIR" \
    bash "$GEAK_ROOT/kernel_workflow/scripts/gpu_lock.sh" 1 "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import flydsl
from flydsl.runtime.device import get_rocm_arch
import torch

target = Path(os.environ["TARGET_DIR"])
cases = json.loads((target / "baseline/cases.json").read_text())["cases"]
roles = Counter(item["coverage_role"] for item in cases)
assert len(cases) == 66, len(cases)
assert sum(bool(item["weight"]) for item in cases) == 58
assert {item["gqa_ratio"] for item in cases} == set(range(1, 17))
assert {item["head_size"] for item in cases} == {128, 256}
assert {item["query_dtype"] for item in cases} == {"bfloat16", "float16"}
assert {item["kv_dtype"] for item in cases} == {
    "fp8", "fp8fnuz", "bfloat16", "float16"
}
assert roles["seed_flydsl_guard"] == 2
assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
arch = str(get_rocm_arch())
assert arch.startswith("gfx1201"), arch
print(
    f"RDNA4 SplitKV full-fallback preflight: {props.name}, {arch}, "
    f"cases={len(cases)}, scored=58, physical_gpu=1"
)
PY

if [ "${GEAK_PREFLIGHT_ONLY:-0}" = "1" ]; then
    bash "$GEAK_ROOT/kernel_workflow/scripts/gpu_lock.sh" 1 \
        "$PYTHON_BIN" "$TARGET_DIR/baseline/test_kernel.py" \
        --case score_fp8_bf16_d256_g1_p16_s257 \
        --case guard_seed_fp8_bf16_d256_g6_p1568_s8192 \
        --case score_bfloat16_bfloat16_d256_g4_p32_s2048 \
        --case score_fp8_float16_d128_g8_p32_s4097 \
        --case score_fp8fnuz_bfloat16_d256_g4_p128_s2048 \
        --case score_ragged_b3_fp8_bfloat16_d256_g5 \
        --draws 1 --correctness-only
    echo "GEAK full-fallback oracle preflight passed; workflow not started."
    exit 0
fi

cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
