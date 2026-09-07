# Qwen3.8-27B FP8-KV SplitKV — RDNA4 GEAK target

This directory is a runnable GEAK target around vLLM's ROCm paged-decode SplitKV
path. The frozen seed incorporates FP8 K/V scales and exposes complete stage-1 plus
reduction timing with caller-owned scratch buffers.

The checked-in campaign uses a 16-direction budget and physical GPU 1 only. GEAK's
`gpu_ids="1"`, allocation fence, filesystem lock, idle check, and per-command
`HIP_VISIBLE_DEVICES=1` selector all use the same host-physical ID. Inside each
pinned subprocess, PyTorch sees that card as logical device 0.

## Start GEAK

First verify that Codex is authenticated, then start the campaign:

```bash
cd /app/GEAK
codex login status
bash targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/run_geak.sh \
  |& tee targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/capture/geak.log
```

GEAK counts optimization directions rather than promising one process-level round
per budget unit. The task asks the Tech Lead to select one direction per round, so
the 16-direction budget normally yields 16 interpretable rounds. It also sets the
stall allowance to 16 so the usual early-stall cutoff does not truncate the campaign.

The run writes timestamped results under `/app/GEAK/exp/`. It does not apply the
winner to `/app/vllm` automatically (`apply_to_original=false`).

To validate the GPU fence and inputs without starting GEAK, use
`GEAK_PREFLIGHT_ONLY=1 bash targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/run_geak.sh`.

## Frozen optimization surface

- `baseline/kernel.py`: editable FP8 SplitKV seed copied into each GEAK workspace.
- `baseline/test_kernel.py`: paged-layout input builder, independent vLLM 2D seed
  check, and complete-operation CUDA-event benchmark.
- `baseline/cases.json`: captured guards plus long-context/ragged optimization cases.
- `baseline/workload.json`: scoring and hardware contract.
- `baseline/EXPERIMENT_CONTEXT.md`: mandatory prior evidence, constraints, and
  optimization directions.

Run a quick preflight on physical GPU 1 with:

```bash
cd /app/GEAK/targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/baseline
GEAK_GPU_ALLOWED=1 PYTHONPATH=/app/vllm \
  bash /app/GEAK/kernel_workflow/scripts/gpu_lock.sh 1 \
  /opt/python/bin/python test_kernel.py \
  --case split_b1_s8192 --check --warmup 2 --repeats 5
```

The long-context cases use equal weights as an explicit target prior; they are not
presented as measured production frequencies. Captured contexts up to 4014 tokens
and the B30 ragged scheduler shape remain zero-weight hard regression gates.

## Runtime capture provenance

The capture tooling establishes the real operator contract without editing the vLLM
checkout. A `sitecustomize` import hook records tensor metadata at
`chunked_prefill_paged_decode`, and the offline workload runs Qwen3.8-27B-FP8 with
TP2 and `kv_cache_dtype="fp8"` on two R9700s.

## Capture

```bash
cd /app/GEAK
bash targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/run_capture.sh
```

The default vLLM environment is `/opt/python/bin/python`; override it with
`PYTHON_BIN`. Other useful overrides are `VLLM_SRC`, `MODEL`, `REVISION`, `TP`,
`MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, and `CAPTURE_GPUS`.

Summarize an existing capture with:

```bash
/opt/python/bin/python \
  targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/summarize_capture.py \
  --shape-log \
  targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/capture/raw_calls.jsonl \
  --output \
  targets/qwen3_8_27b_fp8_kv_splitkv_rdna4/capture/summary.json
```

The checked-in [manifest](capture/manifest.json) and
[summary](capture/summary.json) retain the reproducible facts. Raw calls and logs are
ignored because they contain redundant per-layer/per-rank records.

See [EXPLORATION.md](EXPLORATION.md) for the discovery-stage evidence and the larger
design space behind the frozen campaign contract.
