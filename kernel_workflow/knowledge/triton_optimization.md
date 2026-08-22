# Triton Kernel Optimization Patterns

Patterns ranked by priority. Higher priority (P0) = higher expected impact.

Run `scripts/detect_gpu_arch.sh` first. Read `amd_rdna4.md` for gfx120x and
`amd_instinct.md` for CDNA. Triton's AMD `num_warps` follows the compiled target:
wave32/WMMA on RDNA4 versus wave64/MFMA on CDNA.

## P0: Algorithm & Tiling Design

### Tiling Strategy
Triton kernels are fundamentally block-based. The tiling scheme determines performance more than anything else.

**Key decisions:**
- Choose block dimensions that maximize data reuse
- Start with BLOCK_SIZE multiples of 64 for portability; also test 32 for naturally tiny RDNA4 work
- Balance tile size vs register pressure vs shared memory usage

```python
@triton.jit
def kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    # Process full block at once
    y = tl.exp(x)
    tl.store(Y_ptr + offsets, y, mask=mask)
```

### Reduction Patterns
For reductions, use hierarchical approach: per-block reduction → cross-block atomic or two-pass.

```python
@triton.jit
def reduce_kernel(X_ptr, OUT_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.atomic_add(OUT_ptr, result)
```

### Multi-Dimensional Tiling
For 2D problems (matmul, attention), tile both dimensions independently.

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Accumulate over K in BLOCK_K tiles
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + ...)  # BLOCK_M x BLOCK_K tile
        b = tl.load(B_ptr + ...)  # BLOCK_K x BLOCK_N tile
        acc += tl.dot(a, b)       # Matrix-core lowering: MFMA on CDNA, WMMA on RDNA4
    tl.store(C_ptr + ..., acc)
```

### Fused Operations
Combine multiple elementwise operations into a single kernel pass to reduce memory traffic.

## P1: Memory Access Patterns

### Coalesced Block Loading
Ensure `tl.load` accesses contiguous memory within each block. Use `tl.arange` with stride 1.

```python
# GOOD: Contiguous access
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offsets)

# BAD: Strided access
offsets = tl.arange(0, BLOCK) * stride  # Non-contiguous
```

### Block Size Selection
- Minimum useful full-wave tile: 32 on RDNA4, 64 on CDNA
- Sweet spots: 64, 128, 256, 512, 1024
- Larger blocks = more data reuse but more register pressure

### Masking
Always use masks for boundary conditions. Avoid `tl.where` when possible — it generates unnecessary instructions.

```python
# Prefer mask parameter over tl.where
x = tl.load(ptr + offsets, mask=mask, other=0.0)  # Faster
# Avoid: x = tl.where(mask, tl.load(ptr + offsets), 0.0)
```

## P2: Compute Optimization

### Constexpr Hints
Mark values known at compile time as `tl.constexpr` to enable compiler optimizations.

```python
@triton.jit
def kernel(N, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    # Compiler can unroll loops with constexpr bounds
    for i in range(NUM_STAGES):  # Unrolled
        ...
```

### Dot Product → Matrix Core
On AMD, `tl.dot` can map to MFMA on CDNA or WMMA on RDNA4. Use it for matrix operations and inspect
the generated ISA to prove the intended lowering.

- Input types: fp16, bf16, fp32, int8
- Minimum sizes: typically 16x16 tiles
- Returns fp32 accumulator

### Mixed Precision
Load in fp16/bf16, compute in fp32 for bandwidth savings with precision.

```python
x = tl.load(ptr + offsets).to(tl.float16)  # Load as fp16
acc += tl.dot(x, y)  # Compute in fp32 via the target's matrix core
```

## P3: AMD-Specific Optimizations

### waves_per_eu
Control occupancy via the `waves_per_eu` parameter in `@triton.autotune`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2,
                      waves_per_eu=2),  # AMD-specific
    ],
    key=['N'],
)
```

### Target wave size
- CDNA gfx942/gfx950 uses wave64: four-byte contiguous lanes cover 256 bytes.
- RDNA4 gfx120x uses native wave32: four-byte contiguous lanes cover 128 bytes.
- `num_warps`, reduction depth, register pressure, and threads/program therefore change across families.

### Matrix-core tile sizes
Match `BLOCK_M/N/K` to the target matrix instruction shapes (detect the arch with
`rocminfo`):
- **gfx942 (CDNA3)**: 4x4x4, 16x16x16, 32x32x8 (plus 16x16x32 / 32x32x16 for 8-bit). Prefer
  `matrix_instr_nonkdim=16`.
- **gfx950 (CDNA4)**: adds new/wider MFMA variants and native MXFP4/MXFP6/MXFP8 (block-scaled) matrix
  ops not present on gfx942 — a major low-precision GEMM lever. See `amd_instinct.md` §3.
- **gfx1200/gfx1201 (RDNA4)**: wave32 WMMA, with a base 16x16x16 FP16/BF16→FP32 operation and a
  gfx12-specific fragment ABI. Do not use `matrix_instr_nonkdim` or MFMA assumptions; see `amd_rdna4.md`.

## P4: Autotune Configurations

### @triton.autotune
Define multiple configurations and let Triton pick the fastest.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16},
                      num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],  # Re-tune when these change
)
@triton.jit
def kernel(M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

### Key Selection
Choose autotune keys that capture shape-dependent behavior. Include dimensions that affect tiling efficiency.

## Compound Strategy Compatibility

Compose well:
- Tiling + target matrix-core dot → excellent (standard matmul pattern)
- Fused ops + Coalesced loading → excellent
- Autotune + Multiple tile sizes → excellent (let runtime decide)
- Mixed precision + matrix core → excellent when ISA inspection proves native lowering

Conflicts:
- Very large tiles + High num_warps → register pressure
- Many num_stages + Large tiles → shared memory overflow
