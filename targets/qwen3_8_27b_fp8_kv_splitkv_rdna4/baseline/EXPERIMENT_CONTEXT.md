# FP8 SplitKV campaign memory

This target optimizes the complete two-dispatch SplitKV decode operation for the full-attention
layers of Qwen3.8-27B-FP8 under tensor parallelism 2 on `gfx1201`.

## Immutable contract

- Query/output: BF16 `[B, 12, 256]`.
- KV cache: FP8 E4M3, two KV heads, GQA ratio 6.
- Physical page size: 1568 tokens.
- K: `[blocks, 2, 16, 1568, 16]`, stride `[1605632,401408,25088,16,1]`.
- V: `[blocks, 2, 256, 1568]`, stride `[1605632,401408,1568,1]`.
- Apply arbitrary FP32 K/V scales after FP8 load and before BF16 dot products.
- Time stage 1 and reduction together with caller-owned output and FP32 scratch.

The input kernel is the correctness and timing denominator. vLLM's current stride-aware 2D FP8
kernel is only an independent seed smoke check; candidates must be compared to the immutable
SplitKV baseline created by the freezer.

## Evidence already collected

- The real model capture produced B=1/3/13/30 decode geometries at D=256, Hq=12, Hkv=2.
- A minimal FP8 SplitKV prototype was correct at non-unit scales against the 2D kernel.
- On one R9700 with B=1 and 14 forced splits, isolated synthetic measurements showed 8.81x at
  sequence 4014 and 14.41x at sequence 32768 versus the incumbent 2D kernel. These were viability
  measurements, not end-to-end serving results.
- The upstream occupancy heuristic does not engage SplitKV below roughly 4096 tokens for this GPU;
  scheduling is therefore part of the optimization surface, not a fixed truth.

## Workload interpretation

The nonzero-weight cases are an explicit equal-weight long-context optimization prior. They are not
claimed production frequencies. Captured short/mid contexts and a large ragged batch are zero-weight
hard gates. Optimize the weighted ratio of sums, but retain every case at >=0.98x and correct output.

## Highest-value directions

1. Jointly tune split count and logical compute tile (32/64/128); do not require the tile to divide
   the 1568-token physical page because address calculation already maps each logical token.
2. Move or vectorize FP8-to-FP32 scale application and test cache/eviction policies.
3. Reduce wasted work from padding GQA=6 to 16 rows.
4. Reduce FP32 scratch and LSE reduction traffic while preserving stable softmax composition.
5. Specialize decode control flow and precompute safe launch metadata on the host.
6. Consider fusion only after measuring the complete two-dispatch cost; prior evidence is mixed.

Never memoize outputs or operands, reduce the case set, time only stage 1, use unit-only scale tests,
or introduce exact-shape tables that do not generalize to interior sequence lengths.
