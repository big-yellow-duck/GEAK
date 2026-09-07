#!/usr/bin/env python3
"""Run a small vLLM offline workload and capture block-FP8 linear calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prompt_with_words(words: int, index: int) -> str:
    stem = f"Request {index}: explain one useful GPU optimization principle. "
    return stem + ("AMD matrix kernel shape capture " * max(1, words // 5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--revision", default="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")
    p.add_argument("--shape-log", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = p.parse_args()

    shape_log = str(Path(args.shape_log).resolve())
    os.environ["GEAK_W8A8_SHAPE_LOG"] = shape_log
    os.environ["GEAK_W8A8_CAPTURE_PHASE"] = "offline_mixed"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

    # Import only after setting the worker-inherited capture environment.
    from vllm import LLM, SamplingParams

    workloads = [
        {"name": "decode_b1", "batch": 1, "prompt_words": 64, "max_tokens": 2},
        {"name": "decode_b4", "batch": 4, "prompt_words": 128, "max_tokens": 2},
        {"name": "prefill_512", "batch": 2, "prompt_words": 512, "max_tokens": 1},
        {"name": "prefill_1024", "batch": 1, "prompt_words": 1024, "max_tokens": 1},
    ]

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=True,
        # Qwen3.5/3.8 is a hybrid vision-language architecture. This target is
        # the text linear workload, so do not load/profile the unused encoder.
        language_model_only=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=max(w["batch"] for w in workloads),
        max_num_batched_tokens=max(args.max_model_len, 4096),
        gpu_memory_utilization=args.gpu_memory_utilization,
        kernel_config={"linear_backend": "triton"},
        disable_log_stats=False,
        seed=17,
    )

    # Engine initialization performs synthetic memory profiling at M=4096.
    # The target is the real offline prompt mix, so discard those warmup calls
    # only after all workers are initialized and idle.
    Path(shape_log).write_text("")

    completed = []
    for wi, workload in enumerate(workloads):
        prompts = [
            prompt_with_words(workload["prompt_words"], wi * 10 + i)
            for i in range(workload["batch"])
        ]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=workload["max_tokens"],
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, params, use_tqdm=False)
        completed.append(
            {
                **workload,
                "prompt_token_counts": [len(o.prompt_token_ids) for o in outputs],
                "generated_token_counts": [
                    len(o.outputs[0].token_ids) for o in outputs
                ],
            }
        )
        print("GEAK_CAPTURE_WORKLOAD " + json.dumps(completed[-1], sort_keys=True))

    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()

    manifest = {
        "schema": "geak.w8a8_blockscale_capture.v1",
        "model": args.model,
        "revision": args.revision,
        "hardware": {
            "gfx": os.environ.get("GEAK_GPU_GFX", ""),
            "arch_class": os.environ.get("GEAK_GPU_ARCH_CLASS", ""),
            "wave_size": int(os.environ.get("GEAK_GPU_WAVE_SIZE", "0") or 0),
            "physical_cu_count": int(os.environ.get("GEAK_GPU_CU_COUNT", "0") or 0),
            "wgp_count": int(os.environ.get("GEAK_GPU_WGP_COUNT", "0") or 0),
        },
        "tensor_parallel_size": args.tp,
        "language_model_only": True,
        "linear_backend": "triton",
        "max_model_len": args.max_model_len,
        "workloads": completed,
        "shape_log": shape_log,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print("GEAK_CAPTURE_MANIFEST " + str(Path(args.manifest).resolve()))


if __name__ == "__main__":
    main()
