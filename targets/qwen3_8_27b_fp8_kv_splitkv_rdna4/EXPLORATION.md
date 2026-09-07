# FP8 SplitKV exploration for Qwen3.8-27B on gfx1201

## HIP authoring campaign decision

The original Triton exploration is complete. The current campaign keeps that
implementation frozen as the live oracle and performance denominator, then:

1. Authors a fresh native-HIP gfx1201 implementation supporting FP8 E4M3FN and
   BF16 KV caches over the same stride-padded 1,568-token page contract.
2. Gives the HIP lane 16 optimization directions and accepts a winner only when
   it beats the frozen Triton SplitKV on the weighted FP8 workload while passing
   every FP8 and BF16 correctness/performance guard.

Do not run GEAK directly against PR #2 as-is. The PR only widens the dispatch from
head size 256 to `{128, 256}`. Its SplitKV kernel has no `k_scale` or `v_scale`
inputs, and the wrapper rejects every FP8 KV cache through
`"fp8" not in kv_cache_dtype`.

Source pins used in this exploration:

- Upstream SplitKV work: `vllm-project/vllm#45916`; fork branch snapshot
  `feiyehua/rocm-gfx12xx-splitkv`, commit `2d08f7f`.
- Head-size widening PR: `feiyehua/vllm#2`, commit `983ea71`.
- Model: `Qwen/Qwen3.8-27B-FP8`, revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Capture hardware: 2 × Radeon AI PRO R9700, `gfx1201`, 32 WGPs per GPU.

The duplicate-work check found that the base SplitKV implementation is already the
open upstream PR #45916 and is tracked by performance issue #50264. Its author
explicitly lists FP8 as unsupported and says precision expansion is planned. Any
integration from this campaign should therefore be coordinated as an extension of
#45916 (or its branch), not opened as a competing standalone PR. The GEAK target and
its receipts remain useful because neither #45916 nor fork PR #2 covers FP8 KV
storage, Qwen3.8's 1,568-token pages, or GQA6 TP2 geometry.

## Model geometry

Qwen3.8 has 64 language layers in a 3:1 linear/full-attention pattern, so only 16
layers call paged full attention. Full attention has 24 query heads, 4 KV heads,
and head size 256. At TP2 the per-rank kernel geometry is therefore:

| Field | Value |
|---|---:|
| Query heads | 12 |
| KV heads | 2 |
| Queries per KV head | 6 |
| Head size | 256 |
| Query/output dtype | BF16 |
| KV storage dtype | FP8 E4M3FN |
| FP8 K packing width `x` | 16 |

This means PR #2's head-size-128 change is not on the hot path for this model.
Head-size 128 should remain in general regression coverage, but the GEAK score must
focus on head-size 256, GQA ratio 6, and two local KV heads.

## Captured operator contract

The live TP2 capture used `kv_cache_dtype="fp8"`, eager execution, a 4,096-token
limit, and the ROCm attention backend. It recorded 608 full-attention calls across
two ranks. Each decode signature appears 32 times: 16 full-attention layers × 2
ranks.

The cache layout was invariant across all calls:

| Tensor | Shape | Stride |
|---|---|---|
| K | `[543, 2, 16, 1568, 16]` | `[1605632, 401408, 25088, 16, 1]` |
| V | `[543, 2, 256, 1568]` | `[1605632, 401408, 1568, 1]` |

Important consequences:

- The physical attention block is 1,568 tokens because hybrid KV allocation must
  cover the Gated DeltaNet state page.
- K and V each occupy half of a shared physical page. Their block stride is twice
  their logical dense extent, so `has_native_kv_cache_layout` is false.
- The compute tile can remain smaller than the physical page; addressing must use
  the captured strides and per-token physical-block mapping.
- `x` must be derived from cache element width. FP8 uses `x=16`; the PR tests use
  BF16 and hardcode `x=8` in their fixture.
- The checkpoint currently supplies scalar K/V scales of 1.0, but the call does not
  assert `unit_kv_scale`. Correctness cases must use randomized non-unit K and V
  scales so an accidental scale specialization cannot pass.

Captured decode launches were:

| Query shape | Max KV length | Block table |
|---|---:|---|
| `[1, 12, 256]` | 180 | `[1, 4]` |
| `[1, 12, 256]` | 1,038 | `[1, 4]` |
| `[1, 12, 256]` | 1,374 | `[1, 4]` |
| `[1, 12, 256]` | 4,014 | `[1, 4]` |
| `[3, 12, 256]` | 1,038 | `[3, 4]` |
| `[13, 12, 256]` | 182 | `[13, 4]` |
| `[30, 12, 256]` | 94 | `[30, 4]` |

The non-power-of-two effective batches are real scheduler output and should stay in
the oracle. They prevent tuning only for clean powers of two.

## Minimum correct FP8 seed

The first implementation should make only these semantic changes to the PR head:

1. Add `k_scale` and `v_scale` to `paged_attention_2d_splitkv_decode` and the
   stage-1 Triton kernel.
2. After FP8 loads, dequantize K and V to the query dtype using the same expression
   as the incumbent 2D kernel: FP8 → FP32, multiply the scalar scale, then cast to
   BF16 before each dot.
3. Pass the real scales through the one-split fallback as well as the split path.
4. Remove only the `"fp8" not in kv_cache_dtype` dispatch exclusion. Keep FP8
   output (`output_scale is not None`), ALiBi, sinks, sliding-window attention, and
   non-gfx1x cases on the incumbent path.
5. Preserve stride-based K/V addressing and derive `x` from the real cache view.
6. Keep partial outputs and LSE in FP32 for the correctness seed.

`kv_cache_dtype="fp8"` on ROCm maps to E4M3FN in this capture. E5M2 is not a
target requirement on RDNA4 and should not broaden the dispatch until separately
validated.

## Prototype viability result

A temporary, in-memory prototype applied exactly the scale/dequant changes above to
PR commit `983ea71` and compared it with vLLM's current FP8 2D Triton kernel using
the captured Qwen geometry, 1,568-token pages, `x=16`, padded strides, non-unit
scales, and preallocated SplitKV scratch.

| Batch | KV length | Splits | 2D | FP8 SplitKV | Speedup | Max abs diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,014 | 14 | 2,324.33 µs | 263.86 µs | 8.81× | 4.88e-4 |
| 1 | 32,768 | 14 | 18,562.65 µs | 1,288.49 µs | 14.41× | 1.22e-4 |

These numbers establish viability, not a production claim. They are isolated
GPU-event medians on one R9700, use synthetic random KV data, force 14 splits, keep
scratch allocation outside timing, and do not include model-level scheduling or
CUDA graphs. The large gain is plausible because the 2D baseline launches only two
workgroups at batch 1 while SplitKV exposes 28, but it must be reproduced in the
frozen target and end-to-end serving.

## GEAK target design

### Source and mode

Use GEAK bakeoff mode with the self-contained Triton SplitKV as the immutable
oracle and request the HIP backend. The incumbent Triton lane remains the floor;
the HIP lane enters author mode, establishes a correct native seed, and then runs
the 16-direction optimization loop. AITER, CK, and FlyDSL are outside this campaign.

The timed callable must include both stage 1 and reduction. Preallocate `mid_out`
and `mid_lse` in the isolated harness so allocator noise is not mistaken for kernel
speed. Separately test the production wrapper under eager mode and CUDA graphs so
scratch lifetime remains an integration gate.

### Immutable oracle

Store a `reference_io.pt` because paged block tables, aliased/padded K/V views, and
ragged sequence metadata are not safely reconstructible from shapes alone. Preserve:

- shared K/V backing storage and the half-page offsets;
- the exact captured strides and FP8 E4M3FN dtype;
- block-table permutations rather than only monotonic physical IDs;
- `seq_lens`, `query_start_loc`, physical block size, and scalar K/V scales; and
- BF16 query/output tensors with GQA ratio 6.

Correctness truth should be independent of the optimized seed: compare with the
incumbent FP8 2D kernel and retain a small PyTorch FP32 attention reference for tiny
cases. The frozen seed can remain the fast live baseline used for candidate timing.

### Case matrix

Retain all seven captured signatures, then add zero-weight or explicitly balanced
generalization sentinels:

- batch `{1, 2, 3, 4, 8, 13, 16, 24, 30, 32}`;
- KV length around `{1, 32, 1,567, 1,568, 1,569, 2,047, 2,048, 2,049,
  4,014, 8,192, 16,384, 32,768, 65,536, 131,072, 262,144}`;
- split count `{1, 2, 4, 7, 9, 14, 16}` including empty final splits;
- head size 256/GQA6 as scored production geometry, with head size 128 as a
  portability guard;
- unit and randomized non-unit K/V scales;
- sequential, permuted, and repeated block-table entries; and
- mixed query batches where `filter_by_query_len` must leave prefill rows untouched.

Do not invent production weights from this exploration workload. Before the final
GEAK run, collect a representative serving trace and score with the count-weighted
ratio of sums. Until then, report every case and use balanced batch/context buckets.

### Performance protocol

- Lock one R9700 and reject foreign GPU activity.
- Compare the two-dispatch callable with cold-cache, reversed-order paired runs;
  long-context working sets must exceed cache capacity.
- Report stage 1 and reduce separately, but gate on their sum.
- Record chosen compute tile, split count, workgroup grid, scratch bytes, VGPRs,
  occupancy, and HBM read bytes per case.
- Re-run accepted candidates under CUDA graphs and in TP2 Qwen serving.
- Gate decode time-to-next-token and throughput; prefill must remain on its existing
  path and cannot regress.

### Acceptance gates

- All randomized and captured cases match the incumbent FP8 2D output at the
  established BF16 attention tolerances, with no NaN/Inf.
- Every non-unit-scale case passes, including independent K and V scales.
- No out-of-range cache read occurs when the last physical page is partial or when
  a split has no tokens.
- Every case stays at least 0.98× the better of the FP8 SplitKV seed and 2D fallback;
  the scored ratio of sums must improve by at least 2% over the frozen SplitKV seed.
- CUDA-graph replay performs no host synchronization or steady-state allocation.
- TP2 model output parity and a model-level quality check pass with
  `--kv-cache-dtype fp8`.

## Highest-value GEAK directions

1. **Split scheduling and tile size.** The current heuristic was calibrated for
   BF16 caches and picks 14 splits for batch 1 but has non-monotonic choices such as
   7 splits at batch 13 and 4 at batch 24. Jointly sweep compute tiles 32/64/128 and
   split counts against `(batch × KV heads × splits)` occupancy and scratch cost.
   The address calculation already supports a tile crossing a physical page, so
   compute tile size need not be restricted to a divisor of 1,568.
2. **FP8 load/dequant path.** Hoist scalar scale loads, verify vectorized `x=16`
   cache transactions, and re-sweep cache hints for 8-bit elements. Preserve BF16
   dots and FP32 accumulation unless a lower-precision alternative passes the full
   oracle.
3. **GQA6 row waste.** The PR pads six queries per KV head to 16 rows, so ten rows
   per group are masked. Explore legal smaller dot shapes or a mapping that reduces
   padded rows without losing the SplitKV workgroup count.
4. **Scratch/reduction traffic.** Measure FP32 partial-output cost before trying
   BF16 partials, alternate normalized/unnormalized partial protocols, or different
   reduce launch geometry. Head size 256 makes scratch material at high split
   counts.
5. **Decode-specialized control flow.** Promote `max_query_len == 1` and exact
   decode-grid facts to compile-time choices, removing mixed-query filtering from
   the hot production variant while retaining the general fallback.
6. **Wrapper scratch lifecycle.** Move intermediate buffers into reusable attention
   workspace or graph-owned storage once kernel tuning is stable. Measure eager and
   graph paths separately.

Do not prioritize cross-workgroup single-dispatch fusion in the first campaign.
GEAK's existing evidence is architecture- and protocol-dependent: naive global
fences can serialize the grid, while a carefully designed HIP arrival protocol can
win. Price the reduction dispatch first and revisit fusion only if its measured
ceiling justifies the added correctness risk.
