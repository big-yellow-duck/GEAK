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
# The first gfx agent is sufficient on the homogeneous GPU boxes GEAK normally
# uses. Callers on a heterogeneous box should set GEAK_GPU_GFX explicitly.

set -euo pipefail

gfx="${GEAK_GPU_GFX:-}"
cu_count="${GEAK_GPU_CU_COUNT:-}"
rocminfo_text=""
if [ -z "$gfx" ] || [ -z "$cu_count" ]; then
    rocminfo_text="$(rocminfo 2>/dev/null || true)"
fi
if [ -z "$gfx" ]; then
    gfx="$(printf '%s\n' "$rocminfo_text" | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
fi
if [ -z "$cu_count" ] && [ -n "$gfx" ]; then
    # rocminfo uses physical RDNA CU terminology (R9700 = 64). HIP/PyTorch's
    # multi_processor_count uses the WGP scheduling count instead (R9700 = 32).
    cu_count="$(printf '%s\n' "$rocminfo_text" | awk -v gfx="$gfx" '
      $1 == "Name:" && $2 == gfx { in_agent=1; next }
      in_agent && $1 == "Compute" && $2 == "Unit:" { print $3; exit }
    ')"
fi

case "$gfx" in
    gfx942)         arch_class=cdna3; wave_size=64 ;;
    gfx950|gfx95*)  arch_class=cdna4; wave_size=64 ;;
    gfx1200|gfx1201|gfx120*) arch_class=rdna4; wave_size=32 ;;
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
