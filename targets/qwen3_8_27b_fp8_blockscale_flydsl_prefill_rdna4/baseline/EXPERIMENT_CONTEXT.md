# FlyDSL broad-prefill campaign contract

Read this file before planning any direction. The seed is the current standalone
FlyDSL RDNA4 raw-weight implementation for `M=1..64`, with vLLM's live Triton
block-scaled GEMM retained only as the correct fallback for `M>64`. The campaign
goal is to replace that prefill fallback with a compact family of faster FlyDSL
kernels while preserving the existing decode and small-prefill wins.

## Seed provenance

- FlyDSL checkout: `/app/rdna4_fp8_blockscale_flydsl`
- Git commit: `eed78c6dd93fd297765632861587d9c3be82e0fc`
- Kernel: `kernels/gemm/rdna4_fp8_blockscale.py`
- Kernel SHA-256: `ea5eec5ee3a7d7d5b1bfa1081ef2fa82d25b4f25e139d709942ebfd209bcabd1`
- Shared gfx12 synchronization helper: `kernels/common/gfx12_sync.py`
- Sync-helper SHA-256: `5ceeff9d76d8181ca5279a904dca79c6c5c69a3a55bf4b4e07659e5cb50e3661`
- Tensor launcher SHA-256: `47bf4f068250002e1275956841b7441f46ca89a7cb761178f0544e4d5fec0060`

Do not modify the external FlyDSL checkout or `/app/vllm` during the GEAK run.
Candidate work belongs in `kernel_src/`: add local FlyDSL source there and route
to it from `kernel.py`. The external kernel is a pinned seed dependency and a
read-only implementation reference.

## Immutable operation contract

- Target: Radeon AI PRO R9700, `gfx1201`, RDNA4, wave32, 64 physical CUs and
  32 WGP scheduler/resource units.
- A and B: FP8 E4M3FN. A is contiguous `[M,K]`. B is logical `[N,K]`, has
  `stride(1)==1`, and may have an arbitrary padded row stride.
- Activation scales: contiguous FP32 `[M,K/128]`.
- Weight scales: contiguous FP32 `[N/128,K/128]`.
- Output: fresh contiguous BF16 `[M,N]` storage reflecting current inputs.
- N and K are positive multiples of 128.
- Complete every K=128 dot product in FP32, apply that block's independent A
  and B scales in FP32, accumulate scaled blocks in FP32, then cast once.
- One device dispatch per call. No result memoization, activation-dependent
  caching, persistent output, extra reduction dispatch, or host synchronization.
- Raw checkpoint weights only. No preshuffle, persistent weight copy, or
  model-load layout conversion in this campaign.
- Use standalone upstream FlyDSL and gfx120x WMMA. Never use AITER, CK, CDNA
  MFMA, gfx1250-only WMMA_SCALE/TDM/cluster features, or modify Triton.

## Seed micro-routing to preserve

The existing FlyDSL implementation has seven compact routes:

1. split-K M=1;
2. split-K M=2 for N<=8192;
3. packed M=4 for N<=8192;
4. packed M<=16;
5. tiled M=17..32;
6. paired-N M=33..64 for wide N;
7. paired-N M=33..64 for ordinary N.

The manifest contains direct guards for each route plus captured M=1/M=2 Qwen
cases and K=128/384/640/2176 decode sentinels. Keep every guard at least 0.98x
of the frozen seed and retain one-dispatch behavior.

The M1/M2 LDS reduction uses `kernels.common.gfx12_sync`:
`lds_wait(outstanding)`, `lds_fence_signal(outstanding)`, and
`lds_fence_wait()`. Generated gfx1201 ISA retains `s_wait_dscnt`, removes the
generic barrier's unnecessary `global_inv`, and schedules address work between
signal and wait. ATT showed the LDS fence is not the dominant bottleneck;
global/scale-load waits dominate. Reuse the helper for fine-grained LDS
pipelines instead of reintroducing `gpu.barrier()`.

## Shape and scoring design

`cases.json` contains five surfaces:

- 35 scored prefill cases captured from Qwen3.8-27B-FP8 at M=72, 138, 139,
  249, 277, 523, and 784 across all five production `(N,K)` families;
- captured decode and existing FlyDSL-route regression cases;
- Qwen-family boundary and full-cross-section cases covering M=65 through 1024;
- non-Qwen portability cases at `(N,K)=(128,128)` and `(1024,2176)` with odd
  padded B strides;
- K%128 decode guards for M=1/M=2.

The primary metric is the captured-prefill counted ratio of sums. All
zero-weight cases are hard correctness and per-case performance gates; they
cannot be traded away for a production-shape win. Report both primary speedup
and FlyDSL route coverage. A fallback result is not a FlyDSL optimization.

Exact captured-M or exact `(M,N,K)` lookup tables are invalid. Route by compact,
explainable tile-count buckets such as M range/tail behavior, N tile count, and
short/medium/long K. Every claimed bucket must include neighboring and portable
guards. Unlisted valid multiples of 128 must retain a correct fallback.

## Productive 32-direction search space

Use the full direction budget unless correctness or infrastructure blocks the
campaign. Prefer one interpretable mechanism per direction and keep measured
dead ends in the ledger. High-value groups include:

1. M=65..128: BM16/BM32 row tiles, BN64/BN128, two/four/eight waves, tail masks.
2. M=129..256: row-tile count versus grid fill and A-fragment reuse.
3. M=257..512 and M=513..1024: larger macro tiles without excess VGPR pressure.
4. Short/medium/long K policies, including 17 scale blocks at K=2176.
5. Cooperative raw-B loading and LDS layouts that reuse B across multiple A
   rows while respecting arbitrary B row stride.
6. Ping-pong LDS or register prefetch pipelines with explicit
   `lds_wait(outstanding)` distances.
7. WMMA issue/load scheduling, scale-load prefetch, K-loop unroll depth, and
   accumulator partitioning.
8. Direct BF16 stores versus an LDS epilogue when it materially improves
   coalescing without adding a dispatch.
9. N-grid ordering and coarse N/K buckets derived from WGP fill, not model IDs.
10. VGPR/LDS occupancy tradeoffs verified from generated ISA and rocprofv3 ATT.

Do not spend directions rediscovering Triton BM32 results or adding host/runtime
shortcuts. This campaign is specifically about native FlyDSL kernel variants.

## Promotion gates

- Five randomized live-baseline comparisons for every case at BF16
  `rtol=0.02, atol=0.0625`, including row-distinct inputs and output independence.
- Every existing FlyDSL seed, decode, boundary, and portability case >=0.98x.
- At least 1.005x reproducible incremental gain before banking a change and a
  final target of >=1.01x on the captured-prefill ratio of sums.
- A banked candidate must add or improve a general FlyDSL routing bucket; source
  inspection or kernel tracing must prove FlyDSL engagement.
- Final timing uses three cache-flushed, reversed-order paired passes plus CUDA
  graph replay. Report instruction count, VGPR/SGPR/LDS, workgroup size, and the
  dominant ATT stalls for each retained kernel family.
