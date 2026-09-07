# AMD RDNA 4 (`gfx1200` / `gfx1201`) Kernel Reference — DETECT FIRST

Use this card when `rocminfo` reports `gfx1200` or `gfx1201`. RDNA 4 is not a
smaller CDNA 4: it has wave32-native SIMD execution, WGP/CU-pair scheduling,
WMMA matrix instructions, GDDR plus Infinity Cache, and different occupancy and
profiling semantics. Do not reuse the wave64/MFMA/MI-series tables from
`amd_instinct.md`.

Primary references:

- AMD RDNA 4 ISA: https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
- AMD GPU specification table: https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- AMD RDNA 4 WMMA guide: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- AMD occupancy guide: https://gpuopen.com/learn/occupancy-explained/
- rocWMMA supported hardware: https://rocm.docs.amd.com/projects/rocWMMA/en/develop/supported-hardware.html
- FlyDSL architecture guide: https://github.com/ROCm/FlyDSL/blob/main/docs/architecture_guide.md
- GEAK FlyDSL RDNA4 capability/authoring card:
  [`../../perf_knowledge/languages/flydsl/rdna4.md`](../../perf_knowledge/languages/flydsl/rdna4.md)

## 0. Detect the actual device

```bash
eval "$(bash "$SKILL_DIR/scripts/detect_gpu_arch.sh")"
printf 'gfx=%s class=%s logical_wave=%s\n' \
  "$GEAK_GPU_GFX" "$GEAK_GPU_ARCH_CLASS" "$GEAK_GPU_WAVE_SIZE"
printf 'physical_cus=%s wgps=%s\n' "$GEAK_GPU_CU_COUNT" "$GEAK_GPU_WGP_COUNT"
rocm-smi --showproductname 2>/dev/null | head
```

The architecture id selects code generation. Record both **physical CUs** and **WGP scheduler units**.
Never infer either from `gfx1201`: several products share the target.

There is a critical API naming mismatch. On the R9700, rocminfo reports `Compute Unit: 64`, matching
AMD's specification table, but HIP/PyTorch exposes `multi_processor_count=32`. That runtime property is
counting WGP-like scheduling units, not physical RDNA CUs. GEAK therefore records `cu_count=64` and
`wgp_count=32`; never silently compare one API's count with the other's threshold.

| Example product | Target | CUs | VRAM | Infinity Cache | L2 |
|---|---:|---:|---:|---:|---:|
| Radeon AI PRO R9700 | gfx1201 | 64 | 32 GiB | 64 MiB | 8 MiB |
| Radeon RX 9070 XT | gfx1201 | 64 | 16 GiB | 64 MiB | 8 MiB |
| Radeon RX 9070 | gfx1201 | 56 | 16 GiB | 64 MiB | 8 MiB |
| Radeon RX 9070 GRE | gfx1201 | 48 | 16 GiB | 48 MiB | 6 MiB |
| Radeon RX 9060 XT | gfx1200 | 32 | SKU-dependent | 32 MiB | 4 MiB |

These are reference values, not launch constants. Partitions, future SKUs, and
firmware can differ; query the box.

## 1. CU, WGP, SIMD, and wave model

- A **WGP** is a CU pair. It is the main scheduling/resource-sharing unit.
- One **CU** is half a WGP and contains **two SIMD32s** sharing a memory path.
  A WGP therefore contains four SIMD32s.
- Native execution is **wave32**. Hardware can execute wave64, but a wave64 VALU
  instruction occupies a SIMD32 over two cycles and consumes more resources.
- HIP/Triton/FlyDSL kernels should assume wave32 only after compiling for the
  RDNA path. Portable HIP must query `warpSize`; host code can use
  `hipDeviceProp_t.warpSize`. Do not use a literal 32 in code shared with CDNA.
- A workgroup must reside within one WGP because all its waves share LDS and
  synchronize there. Grid/fill reasoning starts from reported **WGPs**; physical
  **CUs** remain a separate peak-normalization/device fact. LDS and workgroup
  residency reasoning use **WGP/CU-pair** resource scope.
- A workgroup size of 64/128/256 is a good search set. Multiples of 64 are a
  cross-generation-friendly starting point, but the unit of subgroup logic on
  RDNA4 is 32. A 64-thread group contains two waves, not one.

### Lane operations

- HIP `__shfl*`, `__ballot`, `__any`, and `__all` operate over the compiled
  wave. In the normal RDNA4 path that is 32 lanes and a 32-bit live mask.
- Never implement a reduction with `lane = tid & 63`, a fixed 64-step shuffle,
  or a 64-bit ballot assumption. Use `warpSize`, `__activemask()`, and a
  wave-size-specialized template when source must also run on CDNA.
- `__syncthreads()` remains a workgroup barrier, not a wave barrier.

## 2. Occupancy and CU-fill calculations

RDNA occupancy is constrained jointly by wave slots, VGPRs, LDS, workgroup
shape, and barriers. Do not use the CDNA `VGPRs/thread -> waves/SIMD` table.

For a candidate kernel, collect:

```bash
# HIP compile-time resource remarks (adjust source/options as needed)
hipcc --offload-arch="$GEAK_GPU_GFX" -O3 \
  -Rpass-analysis=kernel-resource-usage kernel.hip -c

# Code object / ISA inspection
llvm-objdump --mcpu="$GEAK_GPU_GFX" -d kernel.co | less
```

Reason in this order:

1. `waves_per_workgroup = ceil(threads_per_workgroup / 32)` for the wave32 path.
2. Use **WGP count** as the first RDNA scheduling/fill floor. Sweep one and two
   workgroups per WGP (32 and 64 groups on a full R9700), then inspect achieved
   occupancy. A physical-CU one-block rule and PyTorch's `multi_processor_count`
   happen to differ by 2x here; name the unit in every calculation.
3. LDS is allocated for the entire workgroup at WGP scope. AMD's device table
   lists 128 KiB for the CU pair, while this R9700's HSA GROUP pool and FlyDSL's
   gfx1201 allocator expose a **64 KiB per-workgroup cap**. Treat 64 KiB as the
   portable launch limit unless the live runtime/compiler proves otherwise;
   never infer allocatable LDS from the marketing total or assume CDNA4's
   160 KiB.
4. VGPR usage is per lane/wave and allocation is granular. RDNA4 supports
   dynamic VGPR allocation, so measure the compiled resource record and achieved
   waves; do not derive residency from MI300/MI350 register tables.
5. Barriers and large multi-wave workgroups can bind even when VGPR/LDS totals
   appear safe. Try fewer waves per workgroup as a paired control.

Useful grid controls:

- Tiny kernels: launch enough independent workgroups to cover detected WGPs,
  then test 2x and 4x that count to price latency hiding.
- Persistent kernels: derive the initial grid from runtime WGP count and sweep
  one vs two workgroups per WGP. Record physical CU count separately for
  throughput/marketing comparisons. Do not hard-code 32 WGPs or 64 CUs.
- Reductions: compare one wave32 per result, two cooperating waves, and a
  multi-workgroup reduction. A CDNA one-wave-per-result kernel becomes two waves
  if copied unchanged at 64 threads.
- Matrix kernels: count output tiles and waves/WG separately. High theoretical
  occupancy can lose when it reduces accumulator reuse or increases GDDR traffic.

## 3. Memory hierarchy and LDS

- gfx1201 products use **GDDR6 plus Infinity Cache**, not MI-series HBM. Never
  compare achieved GB/s with MI300/MI350 HBM nameplate.
- The official device table lists 64 MiB Infinity Cache and 8 MiB L2 on the
  64-CU gfx1201 parts; smaller SKUs vary. Treat both as shared caches and measure
  the working-set transition instead of assuming XCD-local behavior.
- RDNA has no MI XCD topology. Do not apply XCD round-robin/de-interleave or
  chiplet-L2 swizzles from CDNA cards.
- LDS is 32-bank shared memory. Conflict-free indexing still matters, but bank
  behavior must be tested with wave32 access patterns. A padding rule measured
  with a CDNA wave64 tile is not automatically transferable.
- Coalesce global access per wave32. For 4-byte lanes a fully contiguous wave
  touches 128 bytes. This differs from the workflow's old fixed 256-byte
  wave64 statement.
- Build the memory roof from a same-box streaming microbenchmark at the actual
  clock/power state. Consumer/pro cards and PCIe hosts vary too much for a
  single repository constant.

## 4. Matrix cores: WMMA, not CDNA MFMA

gfx120x uses third-generation RDNA **WMMA**. Do not emit CDNA `v_mfma_*`, use
MFMA tile tables, or use `matrix_instr_nonkdim` guidance copied from gfx942/950.

- The base RDNA4 floating-point WMMA operation is wave32 and 16x16x16 for
  FP16/BF16 inputs with FP32 accumulation.
- HIP intrinsic names carry the gfx12 form, for example
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`.
- RDNA4 changed the operand VGPR ABI from RDNA3: each of 32 lanes owns eight
  elements for a 16x16 operand. Do not reuse gfx11 hand-packed fragments.
- Prefer rocWMMA or FlyDSL atoms to raw builtins unless precise fragment control
  is the point of the experiment. Always inspect ISA for `v_wmma*` and confirm
  that the intended datatype did not lower to scalar/vector fallback.
- FP8 is standard OCP-style (`float8_e4m3fn` / `float8_e5m2`), not gfx942 FNUZ.
  Hardware format availability is not proof that a particular ROCm library or
  compiler release lowers the operation; compile, inspect ISA, and parity-test.
- gfx1201 is not gfx1250. Do not use gfx1250-only FP8/FP4/TDM shapes or assume
  gfx1250's larger LDS. FlyDSL's verifier intentionally distinguishes them.

## 5. Backend policy for GEAK

### Supported defaults

- **HIP**: supported when compiled natively with `--offload-arch=gfx1201` (or
  gfx1200). Write wave-size-portable subgroup code or a dedicated wave32 path.
- **Triton AMD**: supported for kernels that compile and pass the immutable
  oracle. Tune wave32 geometry; never inherit the CDNA MFMA/64-lane defaults.
- **FlyDSL main**: supported independently of AITER. Upstream has a gfx120x
  wave32/WMMA lowering, native architecture detection, RDNA GEMM tests, and
  lists Radeon AI PRO R9700/gfx1201 among verified platforms. Prefer direct
  `flydsl` APIs and upstream `kernels/gemm/rdna_f16_gemm.py` patterns. Set
  `FLYDSL_GPU_ARCH=gfx1201` (the GPU lock wrapper does this automatically).
  FP8/BF8 gfx120x atom support is upstream in `ROCm/FlyDSL@3c03e979`; its atom
  tests passed 6/6 and the 26 applicable RDNA GEMM cases passed on the local
  R9700. Treat gfx1200 as compile-supported until a physical gfx1200 receipt exists.
  Release wheels can lag main: target recognition is necessary but only a
  compile/run/parity smoke of the chosen main-branch kernel establishes availability.
  Before authoring or optimizing, read the GEAK FlyDSL RDNA4 card linked above. It separates the
  upstream dense path, the parity-tested small-M FP8 prototype, and the still-open broad-prefill
  work; a generic “gfx1201 supported” badge is not an optimization recipe.

### Disabled by default

- **AITER: no-go on RDNA4 in this workflow.** Do not import it, run its offline
  GEMM tuner, deploy AITER CSVs, or use AITER-hosted FlyDSL wrappers. Current
  RDNA support is experimental and wrong-arch prebuilt modules / CDNA-shaped CK
  selections have produced launch-time segfaults. Direct FlyDSL remains valid.
- **CK/CK-Tile**: not auto-selected. Individual upstream kernels may support
  RDNA4, but many installed serving stacks carry CDNA-specialized instances.
  Allow only an explicit user request followed by a subprocess smoke test.
- **Raw ASM/MFMA**: CDNA assembly is incompatible. A dedicated gfx12 WMMA
  implementation is a new RDNA backend, not a port-by-flag.

Every optional backend probe must run in its own subprocess with a timeout. A
SIGSEGV or missing code object marks only that candidate unavailable; it must not
take down discovery or invalidate the incumbent/Triton/HIP/FlyDSL lanes.

## 6. Triton/FlyDSL tuning checklist

- Establish the actual compiled wave size before interpreting `num_warps`.
- Start workgroup sweeps at 64/128/256 threads, but also try a 32-thread
  one-wave group for naturally tiny work.
- Re-sweep tile sizes: the CDNA winner's accumulator and LDS footprint is not a
  useful default for WMMA.
- Use WGP count for the first persistent-grid/fill threshold; retain physical
  CU count for peak normalization and report both with explicit units.
- Cap LDS against gfx120x, not gfx950.
- For GEMM, confirm `v_wmma*` in ISA. For reductions, confirm shuffle width and
  active mask. For memory kernels, compare bytes against measured GDDR rate.
- Preserve the immutable oracle's dtype/tolerance and test every case. FP8
  lowering and fragment layout mistakes can be fast and consistently wrong.

## 7. Profiling policy

Use `rocprofv3` first on gfx120x. The workflow selects that order automatically.
rocprofiler-sdk supports gfx12 tracing, while rocprof-compute's published
compatibility table does not currently list discrete gfx1200/gfx1201.

When rich counters are unavailable:

1. keep kernel trace, dispatch count, and duration evidence from rocprofv3;
2. inspect compiler resource remarks and ISA;
3. measure bandwidth and WMMA floors with same-box microbenchmarks;
4. mark roofline/bottleneck confidence low rather than substituting MI counters;
5. continue benchmark-driven optimization—the lack of SoL tables is not a run
   blocker.

## Critical rules

1. Detect `gfx`, physical CUs, and WGPs; never conflate rocminfo CUs with
   HIP/PyTorch `multi_processor_count` on RDNA4.
2. wave32 is the RDNA4 kernel model; portable source queries `warpSize`.
3. WMMA is the matrix path; CDNA MFMA advice does not transfer.
4. Use measured GDDR/Infinity-Cache behavior, never MI HBM/XCD constants.
5. Keep FlyDSL independent and native; do not route it through AITER on RDNA4.
6. Treat AITER as unavailable and isolate every optional backend probe.
7. Correctness plus on-box measurement remains the final authority.
