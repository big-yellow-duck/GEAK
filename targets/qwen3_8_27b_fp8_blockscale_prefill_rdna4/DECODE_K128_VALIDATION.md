# RDNA4 K%128 decode synchronization

Date: 2026-08-24

The protected GEAK HIP/rocWMMA decode source is synchronized with vLLM commit
`07d02c352c`. Odd counts of 128-wide scale blocks use a bounded ceil split, so
K=2176 is processed as 9+8 blocks instead of truncating to 8+8. This requires
no padding, copy, persistent storage, or extra dispatch.

## Coverage

- 10 captured Qwen3.8-27B TP2 decode cases
- 8 zero-weight synthetic decode sentinels
- M=1 and M=2
- K=128, 384, 640, and 2176 sentinels
- Five independent input draws per case
- Triton block-scaled GEMM correctness oracle

All 90 comparisons passed BF16 `rtol=0.01`/`atol=0.01`. The maximum absolute
difference was `3.0517578125e-05`.

## Cold-cache timing

Timing used 10 warmups, 51 samples per leg, a 512 MiB cache flush before every
sample, and both execution orders.

| Group | Cases | Minimum speedup | Geomean speedup |
|---|---:|---:|---:|
| Captured decode | 10 | 1.600x | 1.861x |
| Synthetic K%128 decode | 8 | 1.840x | 2.262x |
| All decode gates | 18 | 1.600x | 2.030x |

The captured call-count-weighted decode speedup was 1.799x. At the intended
K=2176 shape, M=1 measured 1.879x and M=2 measured 1.840x against generic
Triton. Every case clears the campaign's 0.98x decode floor.

## Protected sources

- `baseline/binding.cpp` SHA-256:
  `d27cf9138d74b033668c0eaff459db6aae9705118aa4e6b183d1744cf1006538`
- `baseline/kernel.py` SHA-256:
  `b755426fa4616fd50b135b652c5496fd597cffe440fee94ba1a69b7596c048a8`
