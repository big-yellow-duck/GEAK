#!/bin/bash
# Detect the local AMD GPU target and print a stable, shell-friendly contract.
#
# Usage:
#   eval "$(bash detect_gpu_arch.sh)"
#
# Output:
#   GEAK_GPU_GFX=gfx1201
#   GEAK_GPU_ARCH_CLASS=rdna4
#   GEAK_GPU_WAVE_SIZE=32
#   GEAK_GPU_CU_COUNT=64
#   GEAK_GPU_WGP_COUNT=32
#
# Device selection follows gpu_lock.sh: an inherited ROCR allocation is kept;
# otherwise GPU_ID scopes the probe. Explicit gfx/count facts support offline CI.

set -euo pipefail

gfx="${GEAK_GPU_GFX:-}"
cu_count="${GEAK_GPU_CU_COUNT:-}"
if [ -z "$gfx" ] || [ -z "$cu_count" ]; then
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -z "${GPU_ID:-}" ]; then
        rocminfo_text="$(rocminfo 2>/dev/null || true)"
    else
        rocminfo_text="$(ROCR_VISIBLE_DEVICES="$GPU_ID" rocminfo 2>/dev/null || true)"
    fi
    # Read top-level GPU agents only; CPU and nested ISA Name fields are not GPUs.
    facts="$(printf '%s\n' "$rocminfo_text" | awk -v pinned="$gfx" '
      /^ *Agent +[0-9]+ *$/ { in_gpu=0; next }
      /^ *Name: *gfx[0-9a-f]+ *$/ {
        if ($2 != "gfx000" && (pinned == "" || $2 == pinned)) {
          in_gpu=1; current=$2; seen[current]=1
        } else in_gpu=0
        next
      }
      in_gpu && $1 == "Compute" && $2 == "Unit:" {
        if (counts[current] && counts[current] != $3) exit 2
        counts[current]=$3
      }
      END {
        for (arch in seen) { n++; only=arch }
        if (n > 1) exit 2
        if (n == 1) print only, counts[only]
      }
    ')" || { echo "ERROR: mixed GPU identities; mask the selected GPU with ROCR_VISIBLE_DEVICES" >&2; exit 1; }
    read -r detected_gfx detected_cu <<< "$facts"
    gfx="${gfx:-${detected_gfx:-}}"
    cu_count="${cu_count:-${detected_cu:-}}"
fi

case "$gfx" in
    gfx942)         arch_class=cdna3; wave_size=64 ;;
    gfx950|gfx95*|gfx125*)  arch_class=cdna4; wave_size=64 ;;
    gfx1200|gfx1201) arch_class=rdna4; wave_size=32 ;;
    gfx10*|gfx11*)  arch_class=rdna;  wave_size=32 ;;
    gfx9*)          arch_class=cdna_or_gcn; wave_size=64 ;;
    *)              arch_class=unknown; wave_size=0 ;;
esac

wgp_count="${GEAK_GPU_WGP_COUNT:-}"
if [ -z "$wgp_count" ] && [ "$arch_class" = rdna4 ] && [[ "$cu_count" =~ ^[0-9]+$ ]]; then
    wgp_count=$(( (cu_count + 1) / 2 ))
fi

printf 'GEAK_GPU_GFX=%q\n' "$gfx"
printf 'GEAK_GPU_ARCH_CLASS=%q\n' "$arch_class"
printf 'GEAK_GPU_WAVE_SIZE=%q\n' "$wave_size"
printf 'GEAK_GPU_CU_COUNT=%q\n' "$cu_count"
printf 'GEAK_GPU_WGP_COUNT=%q\n' "$wgp_count"
