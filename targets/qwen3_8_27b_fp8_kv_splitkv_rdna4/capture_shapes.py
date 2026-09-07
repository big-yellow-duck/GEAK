#!/usr/bin/env python3
"""Capture the real Qwen3.8 FP8-KV paged-attention call contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def prompt_with_words(words: int, index: int) -> str:
    stem = f"Request {index}: summarize a GPU decode optimization. "
    phrase = "paged attention split kv fp8 cache "
    return stem + phrase * max(1, words // 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument(
        "--revision", default="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    )
    parser.add_argument("--shape-log", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vllm-src", type=Path, default=Path("/app/vllm"))
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    shape_log = Path(args.shape_log).resolve()
    vllm_src = args.vllm_src.resolve()
    vllm_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=vllm_src, text=True
    ).strip()
    vllm_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=vllm_src, text=True
        ).strip()
    )
    attention_source_dirty = (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--",
                "vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
            ],
            cwd=vllm_src,
            check=False,
        ).returncode
        != 0
    )
    os.environ["GEAK_SPLITKV_CAPTURE_LOG"] = str(shape_log)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    from vllm import LLM, SamplingParams

    workloads = [
        {"name": "decode_b1_short", "batch": 1, "prompt_words": 128},
        {"name": "decode_b1_mid", "batch": 1, "prompt_words": 1024},
        {"name": "decode_b1_long", "batch": 1, "prompt_words": 3000},
        {"name": "decode_b4_mid", "batch": 4, "prompt_words": 768},
        {"name": "decode_b16_short", "batch": 16, "prompt_words": 128},
        {"name": "decode_b32_short", "batch": 32, "prompt_words": 64},
    ]

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        trust_remote_code=True,
        language_model_only=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=max(workload["batch"] for workload in workloads),
        max_num_batched_tokens=max(args.max_model_len, 4096),
        gpu_memory_utilization=args.gpu_memory_utilization,
        kernel_config={"linear_backend": "triton"},
        disable_log_stats=False,
        seed=23,
    )

    shape_log.write_text("")
    completed = []
    for workload_index, workload in enumerate(workloads):
        prompts = [
            prompt_with_words(
                workload["prompt_words"], workload_index * 100 + request_index
            )
            for request_index in range(workload["batch"])
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
        row = {
            **workload,
            "prompt_token_counts": [len(output.prompt_token_ids) for output in outputs],
            "generated_token_counts": [
                len(output.outputs[0].token_ids) for output in outputs
            ],
        }
        completed.append(row)
        print("GEAK_SPLITKV_WORKLOAD " + json.dumps(row, sort_keys=True))

    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()

    manifest = {
        "schema": "geak.splitkv_paged_decode_capture.v1",
        "model": args.model,
        "revision": args.revision,
        "vllm_source": {
            "path": str(vllm_src),
            "revision": vllm_revision,
            "dirty": vllm_dirty,
            "attention_source_dirty": attention_source_dirty,
        },
        "tensor_parallel_size": args.tp,
        "kv_cache_dtype": "fp8",
        "language_model_only": True,
        "max_model_len": args.max_model_len,
        "hardware": {
            "gfx": os.environ.get("GEAK_GPU_GFX", ""),
            "arch_class": os.environ.get("GEAK_GPU_ARCH_CLASS", ""),
            "wave_size": int(os.environ.get("GEAK_GPU_WAVE_SIZE", "0") or 0),
            "physical_cu_count": int(os.environ.get("GEAK_GPU_CU_COUNT", "0") or 0),
            "wgp_count": int(os.environ.get("GEAK_GPU_WGP_COUNT", "0") or 0),
        },
        "workloads": completed,
        "shape_log": str(shape_log),
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"GEAK_SPLITKV_MANIFEST {manifest_path}")


if __name__ == "__main__":
    main()
