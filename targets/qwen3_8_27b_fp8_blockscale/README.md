# Qwen3.8-27B-FP8 — vLLM Triton to FlyDSL bake-off

This target captures the real calls to vLLM's `TritonFp8BlockScaledMMKernel` and freezes its
`w8a8_triton_block_scaled_mm` function as the correctness/performance denominator for GEAK. The vLLM
probe is opt-in through `GEAK_W8A8_SHAPE_LOG`; ordinary vLLM runs perform no logging.

## 1. Capture and prepare

```bash
cd /app/GEAK
VLLM_SRC=/app/vllm bash targets/qwen3_8_27b_fp8_blockscale/run_capture.sh
```

The run pins `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, TP=2, vLLM's Triton linear backend, eager execution,
language-model-only mode (the checkpoint is a hybrid architecture), and several decode/prefill prompt
batches. It writes raw calls under `capture/`, aggregates one workload case per exact tensor/layout
signature, benchmarks the Triton baseline, then generates
`bakeoff.args.json`. If the opt-in logging seam is absent from the vLLM checkout, `run_capture.sh`
installs the committed [`vllm_shape_logging.patch`](vllm_shape_logging.patch) automatically; it does
not touch any other vLLM file.

The runner clears initialization-time synthetic calls after the engine becomes ready. Consequently,
the generated call counts and weights describe only the submitted prompt/decode mix, not vLLM's
`M=4096` memory-profile warmup.

Decode scenarios request two output tokens: prefill emits the first, and the second forces one real
decode step at batch 1 or 4. Prefill-only scenarios request one token. This captures each execution
regime without weighting the bakeoff by arbitrary repeated generations.

Useful overrides: `MODEL`, `REVISION`, `TP`, `CAPTURE_GPUS`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`,
`GEAK_GPU_IDS`, `GEAK_BUDGET`, `HF_HOME`, and `RESET_CAPTURE=0` to append another workload.

## 2. Inspect the frozen input

```bash
python3 baseline/test_kernel.py --list
HIP_VISIBLE_DEVICES=0 python3 baseline/test_kernel.py --repeats 20
python3 -m json.tool baseline/workload.json | less
```

The frozen seam starts after dynamic activation quantization: A/B and their block scales are inputs;
bias and reshape remain outside. The math contract is an FP32 accumulation of independently scaled
K-block partial dots, followed by one output cast. The workload also freezes vLLM's padded weight row
strides. Hardware metadata distinguishes the R9700's 64 physical CUs from its 32 WGP scheduler units;
FlyDSL grid/fill reasoning starts from WGP count and targets wave32 WMMA.

## 3. Run GEAK

Install a FlyDSL main-compatible build first, then:

```bash
bash run_bakeoff.sh |& tee capture/bakeoff.log
```

RDNA4 policy keeps AITER disabled. The original Triton lane is always retained, while a standalone
FlyDSL gfx1201 wave32/WMMA lane is authored and compared against exactly the same frozen cases.

## 4. Validation verdicts

The cumulative Triton/rocWMMA candidate, including the round-4 wide M=2 decode specialization, has
been revalidated with fresh paired timing and is the selected winner for the current vLLM integration.
See [`TRITON_RDNA4_VALIDATION.md`](TRITON_RDNA4_VALIDATION.md) for the measurements, routing policy,
FlyDSL comparison, known reporting caveats, and remaining upstream packaging checks. FlyDSL round 3
is disqualified for activation-dependent result memoization; the clean round-4 FlyDSL kernel is
promising but deferred until its weight-layout and full-model cache behavior are deployment-ready.

## 5. Sweep the BM32 prefill crossover

The crossover sweep calls the round-4 BM32 kernel directly, bypassing its current signature router,
and compares it with vLLM's incumbent across every captured `(N, K)` weight shape. The defaults cover
powers of two, captured prefill values, and both sides of the proposed exclusive `32 < M < 784`
interval. A sampled `M` is considered route-safe only when every captured weight shape reaches the
requested minimum speedup.

```bash
bash ../../kernel_workflow/scripts/gpu_lock.sh 0 \
  python3 sweep_prefill_bm32.py \
  --output capture/prefill_bm32_sweep.json
```

Use `--m-values 32,33,64,128,256,512,783,784` for a smaller sweep or repeat
`--shape 5120x3072` to select individual captured weight shapes.
