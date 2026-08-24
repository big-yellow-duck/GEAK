# Prior evidence and search contract

Read this file before proposing a direction. The seed is the cumulative round-4 engineer-1 hybrid,
not vLLM's original generic Triton kernel. Optimize prefill only and compare against this seed.

## Immutable contract

- Device: Radeon AI PRO R9700, `gfx1201`, RDNA4, wave32, 64 physical CUs / 32 WGP scheduler units.
- Operation: FP8 E4M3 A and B, FP32 activation/weight block scales, BF16 output, B stored as N×K with
  captured padded row strides, block size 128×128, no bias.
- Math: accumulate each 128-wide K partial in FP32, apply that partial's independent A/B scales in
  FP32, sum the scaled partials in FP32, then cast once to BF16.
- One logical device dispatch per call. No weight preshuffle or extra persistent model-weight copy.
- Every call must recompute from current inputs and return independent output storage. Input/output
  memoization, activation-keyed result caches, and persistent result buffers are invalid.
- `binding.cpp` is the protected decode implementation. Its current SHA-256 is
  `d27cf9138d74b033668c0eaff459db6aae9705118aa4e6b183d1744cf1006538`; do not edit it.
- Preserve `kernel.py`'s K%128 M=1/M=2 `_get_decode_ext().decode(...)` route byte-for-byte. New work
  belongs in prefill kernels and prefill routing only.
- AITER is unavailable on RDNA4. FlyDSL is deliberately out of scope for this target. Do not spend a
  direction probing either one.

## Seed and validated result

The vendored seed came from:

`exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_1/workspace/kernel_src/`

Seed source hashes:

- `kernel.py`: `405b37a4aad43934d75efd0c368c209f6f30447f9da405c1ba82588adb6866d7`
- `binding.cpp`: `7e4b24d241a6f4166344fce10dcef00ec3ddecdac777c1c706393804f6f6b8e1`

The protected decode source was then synchronized with vLLM commit `07d02c352c`: odd counts of
128-wide scale blocks are divided across the split waves with a bounded ceil split, expanding decode
from K%256 to K%128 without padding or an extra dispatch. Current synchronized source hashes:

- `kernel.py`: `b755426fa4616fd50b135b652c5496fd597cffe440fee94ba1a69b7596c048a8`
- `binding.cpp`: `d27cf9138d74b033668c0eaff459db6aae9705118aa4e6b183d1744cf1006538`

Fresh validation reproduced approximately 1.103x on the captured combined operator workload and
1.018x on prefill alone. The prefill win is routed conservatively:

- `M=523`: BM32×BN128 for N/K = 5120/3072, 5120/8704, 7168/5120, and 8192/5120;
- `M=784`: BM32×BN128 for N/K = 5120/3072, 5120/8704, and 8192/5120;
- every other prefill signature: original vLLM Triton fallback.

Do not delete a seed route unless a three-pass paired result proves the replacement improves the
counted prefill metric and its local cases. The primary denominator for this campaign is the validated
seed itself, not the older vLLM-only baseline.

## What profiling established

The captured prefill workload is compute/WMMA-feed/schedule bound, not GDDR-bandwidth bound. The
original Triton prefill variants used roughly 232–240 VGPR, four wave32 waves, two stages, about 25 KiB
LDS, zero scratch, and native `v_wmma_f32_16x16x16_fp8_fp8`. M=523 and M=784 dominated the earlier
weight, but this target scores all 35 captured prefill signatures using their real call counts.

Prioritize tile geometry, program count versus the 32-WGP device, WMMA operand feed, GROUP_M ordering,
K-loop scheduling, and register/occupancy tradeoffs. Bandwidth-only stories, launch-collapse tricks,
and model-weight preshuffling do not fit the evidence or deployment contract.

Evidence: `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/profiling_summary.md`.

## Successful evidence to carry forward

1. BM32×BN128, four warps, two stages lowered the specialized prefill kernel from roughly 240 VGPR to
   143–144 VGPR with no scratch and produced repeatable wins on the seven seed routes.
2. Direct crossover sweeps showed tile-aligned discontinuities rather than one monotonic M threshold:
   - sampled `M=64, 96, 112, 128` were non-regressing across all five captured N/K families;
   - `M=129` immediately regressed, reaching about 0.86x on the long-K family;
   - sampled `M=225..256` were strong across all five families;
   - adjacent `M=224` and `M=257` regressed to about 0.966–0.967x;
   - 51-sample stress passes reproduced the captured `M=72` and `M=249` route opportunity.
3. A broad `32 < M < 784` route is invalid. Outcomes depend on M tile count and N/K family; examples
   include worst-shape results near 0.912x at M=138, 0.965x at M=277, 0.971x at M=512, and 0.947x at
   M=768.

Raw crossover evidence is checked in under the previous target's `capture/prefill_bm32_*.json` files.
Use it as a prior. Do not rerun the same dense BM32-versus-incumbent sweep as an optimization direction.

## Dead ends: do not repeat without a materially new mechanism

- Round 1 broad/universal Triton rewrite failed to reproduce. Keep shape-family routing and fallback.
- For wide M=523/M=784 work, BM64×BN64 reached about 248 VGPR without a retained full-suite win;
  BM64×BN32 measured about 0.808x; eight warps and three stages regressed.
- Extending the existing BM32 route to M=523,N=17408 looked about 1.074x in isolation but fell to
  about 0.957x under the full-suite control. Exact-shape microbench wins are not sufficient.
- Round 4 attacked M=523,N=17408 with several low-accumulator mappings and all lost:
  BM16×BN128/four waves ≈0.695x, BM16×BN128/two waves ≈0.736x, BM32×BN64 ≈0.797x,
  sequential N strips ≈0.786x, and a four-strip macro tile ≈0.796x.
- Host/runtime tricks are out of scope: this target times one already-JIT-compiled device dispatch.
- FlyDSL round 3's huge result was activation-dependent result replay and is invalid. Round 4 was valid
  but depended on a deployment-unready weight-preshuffle lifetime. Neither route belongs here.

Reports:

- `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_3/engineer_0/report.md`
- `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_0/report.md`
- `exp/bakeoff_baseline_20260822_102440/bakeoff/triton/team_task_20260822_103835_290494_26777/task/round_4/engineer_1/report.md`

## Productive search space

Treat routing and implementation as separate questions:

1. First bank low-risk generalization of the existing BM32 kernel only where complete N/K-family
   evidence supports a tile-aligned bucket. Prefer arithmetic predicates derived from tile count over
   a list of captured exact M values.
2. For uncovered M buckets, tune a small configuration family around BM16/BM32/BM64 and BN64/BN128,
   GROUP_M, two/four warps, and one/two stages. Do not repeat the failed wide-M configurations above.
3. Use the 32-WGP count explicitly: compare program count, waves per workgroup, VGPR pressure, and tail
   tiles at each boundary. A different bucket may justify a configuration that failed at M=523 wide-N,
   but state that mechanism before measuring it.
4. Keep a dimension-aware configuration policy. A compact bucketed table or nearest-M configuration
   selector is acceptable; exact production-signature routing without neighboring evidence is not.
5. A new kernel is retained only after it wins the counted score and passes decode plus heldout floors.

## Required measurement and promotion gates

- Correctness: all manifest cases, five random draws, BF16 `rtol=0.01`/`atol=0.01`, captured strides,
  fresh-output independence, and no source/harness/oracle edits.
- Primary metric: `sum(count × seed_ms) / sum(count × candidate_ms)` over the 35 captured prefill cases.
- Promotion: at least 1.005x versus the current cumulative seed.
- Decode floor: every captured and synthetic K%128 M=1/M=2 case at least 0.98x and protected decode
  source unchanged.
- Heldout floor: every synthetic boundary case at least 0.98x. A result below the floor is a routing
  failure even if its zero score weight leaves the primary number above 1.0.
- Definitive timing: three full paired A/B passes, reverse execution order, identical tensors, explicit
  warmup/synchronization, 512 MiB cache flush outside the timed interval, median reporting, idle GPU,
  and GEAK GPU lock. Report fresh counted times; do not mix frozen weights with live speedup ratios.
