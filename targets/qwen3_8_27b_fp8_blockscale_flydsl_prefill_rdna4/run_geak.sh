#!/bin/bash
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAK_ROOT="$(cd "$TARGET_DIR/../.." && pwd)"
ARGS="${GEAK_ARGS:-$TARGET_DIR/bakeoff.args.json}"
export FLYDSL_SEED_ROOT="${FLYDSL_SEED_ROOT:-/app/rdna4_fp8_blockscale_flydsl}"
export VLLM_SRC="${VLLM_SRC:-/app/vllm}"
PYTHON_BIN="${PYTHON_BIN:-/opt/python/bin/python}"

[ -x "$PYTHON_BIN" ] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[ -s "$ARGS" ] || { echo "Missing GEAK args: $ARGS" >&2; exit 2; }
[ -s "$TARGET_DIR/baseline/cases.json" ] || {
    echo "Missing cases.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -s "$TARGET_DIR/baseline/workload.json" ] || {
    echo "Missing workload.json; run $PYTHON_BIN $TARGET_DIR/build_manifest.py" >&2
    exit 2
}
[ -d "$FLYDSL_SEED_ROOT/build-fly/python_packages/flydsl" ] || {
    echo "Missing built FlyDSL Python package under $FLYDSL_SEED_ROOT/build-fly" >&2
    exit 2
}

check_seed_sha() {
    local expected="$1"
    local path="$2"
    local label="$3"
    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        if [ "${ALLOW_FLYDSL_SEED_DRIFT:-0}" = "1" ]; then
            echo "WARNING: $label seed drift: expected $expected, got $actual" >&2
        else
            echo "$label seed drift: expected $expected, got $actual" >&2
            echo "Review the change, update EXPERIMENT_CONTEXT.md, or set ALLOW_FLYDSL_SEED_DRIFT=1." >&2
            exit 2
        fi
    fi
}

check_seed_sha \
    ea5eec5ee3a7d7d5b1bfa1081ef2fa82d25b4f25e139d709942ebfd209bcabd1 \
    "$FLYDSL_SEED_ROOT/kernels/gemm/rdna4_fp8_blockscale.py" \
    "FlyDSL FP8 kernel"
check_seed_sha \
    5ceeff9d76d8181ca5279a904dca79c6c5c69a3a55bf4b4e07659e5cb50e3661 \
    "$FLYDSL_SEED_ROOT/kernels/common/gfx12_sync.py" \
    "gfx12 synchronization helper"
check_seed_sha \
    47bf4f068250002e1275956841b7441f46ca89a7cb761178f0544e4d5fec0060 \
    "$FLYDSL_SEED_ROOT/kernels/common/tensor_shim.py" \
    "FlyDSL tensor launcher"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLYDSL_SEED_ROOT/build-fly/python_packages:$FLYDSL_SEED_ROOT:$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FLYDSL_SEED_ROOT/build-fly/lib:$FLYDSL_SEED_ROOT/build-fly/python_packages/flydsl/_mlir/_mlir_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GEAK_GPU_GFX="${GEAK_GPU_GFX:-gfx1201}"
eval "$(GEAK_GPU_GFX="$GEAK_GPU_GFX" bash "$GEAK_ROOT/kernel_workflow/scripts/detect_gpu_arch.sh")"
export GEAK_GPU_GFX GEAK_GPU_ARCH_CLASS GEAK_GPU_WAVE_SIZE
export GEAK_GPU_CU_COUNT GEAK_GPU_WGP_COUNT
export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-$GEAK_GPU_GFX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$GEAK_GPU_GFX}"
export GPU_ARCHS="${GPU_ARCHS:-$GEAK_GPU_GFX}"
export GEAK_GPU_ALLOWED="0"

if [ "$GEAK_GPU_ARCH_CLASS" != "rdna4" ]; then
    echo "This target requires RDNA4; detected $GEAK_GPU_GFX ($GEAK_GPU_ARCH_CLASS)." >&2
    exit 2
fi
if [ "$GEAK_GPU_GFX" != "gfx1201" ] && [ "${ALLOW_OTHER_RDNA4:-0}" != "1" ]; then
    echo "Receipts are gfx1201-specific. Set ALLOW_OTHER_RDNA4=1 for a portability campaign." >&2
    exit 2
fi

TARGET_DIR="$TARGET_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import flydsl
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
import torch

target = Path(os.environ["TARGET_DIR"])
cases = json.loads((target / "baseline/cases.json").read_text())["cases"]
roles = Counter(case["coverage_role"] for case in cases)
assert len(cases) == 117, f"expected 117 cases, got {len(cases)}"
assert roles["scored_captured_prefill"] == 35
assert roles["flydsl_seed_route_guard"] == 9
assert roles["heldout_prefill_generalization"] == 35
assert roles["portable_prefill_generalization"] == 20
assert len({(case["M"], case["N"], case["K"]) for case in cases}) == len(cases)
assert all(case["N"] % 128 == 0 and case["K"] % 128 == 0 for case in cases)
arch = str(get_rocm_arch())
assert is_rdna_arch(arch) and arch.startswith("gfx120"), f"wrong FlyDSL target: {arch}"
props = torch.cuda.get_device_properties(0)
print(
    f"FlyDSL broad-prefill preflight: {props.name}, {arch}, cases={len(cases)}, "
    f"scored={roles['scored_captured_prefill']}, physical_cus={os.environ['GEAK_GPU_CU_COUNT']}, "
    f"wgps={os.environ['GEAK_GPU_WGP_COUNT']}"
)
PY

if [ "${GEAK_PREFLIGHT_ONLY:-0}" = "1" ]; then
    "$PYTHON_BIN" "$TARGET_DIR/baseline/test_kernel.py" \
        --case decode_m1_n5120_k3072_bn128_bk128 \
        --case flydsl_seed_route_guard_m64_n5120_k3072_bn128_bk128 \
        --case heldout_prefill_generalization_m65_n5120_k3072_bn128_bk128 \
        --draws 1 --correctness-only
    echo "GEAK preflight-only validation passed; the 32-direction job was not started."
    exit 0
fi

mkdir -p "$TARGET_DIR/capture"
cd "$GEAK_ROOT"
exec node "$GEAK_ROOT/interface/codex_workflow_runner.mjs" \
    --script "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
    --args-file "$ARGS"
