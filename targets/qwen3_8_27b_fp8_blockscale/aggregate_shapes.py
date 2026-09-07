#!/usr/bin/env python3
"""Aggregate vLLM shape JSONL into GEAK cases, workload, and bakeoff args."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

KEYS = (
    "a_shape",
    "b_shape",
    "as_shape",
    "bs_shape",
    "a_stride",
    "b_stride",
    "as_stride",
    "bs_stride",
    "a_dtype",
    "b_dtype",
    "as_dtype",
    "bs_dtype",
    "output_dtype",
    "block_size",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape-log", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--bakeoff-json", required=True)
    p.add_argument("--timings")
    p.add_argument("--decode-max-m", type=int, default=16)
    p.add_argument("--gpu-ids", default="0,1")
    p.add_argument("--budget", type=int, default=8)
    args = p.parse_args()

    records = []
    for line in Path(args.shape_log).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("schema") == "geak.w8a8_blockscale_call.v1":
            records.append(r)
    if not records:
        raise SystemExit(
            "no Triton block-FP8 calls captured; verify --linear-backend triton and instrumentation"
        )

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        key = json.dumps({k: r.get(k) for k in KEYS}, sort_keys=True)
        groups[key].append(r)

    timings = {}
    if args.timings and Path(args.timings).exists():
        timings = json.loads(Path(args.timings).read_text()).get("cases", {})

    cases = []
    for key, rows in groups.items():
        r = json.loads(key)
        if len(r["a_shape"]) != 2 or len(r["b_shape"]) != 2:
            continue
        m, k = map(int, r["a_shape"])
        n, bk = map(int, r["b_shape"])
        if k != bk:
            continue
        block_n, block_k = map(int, r["block_size"])
        regime = "decode" if m <= args.decode_max_m else "prefill"
        sig = f"{regime}_m{m}_n{n}_k{k}_bn{block_n}_bk{block_k}"
        by_rank = collections.Counter(str(x.get("rank") or x.get("pid")) for x in rows)
        call_count = max(by_rank.values())
        latency = float(timings.get(sig, {}).get("median_ms", 0.0))
        cases.append(
            {
                "sig": sig,
                "M": m,
                "N": n,
                "K": k,
                "a_shape": r["a_shape"],
                "b_shape": r["b_shape"],
                "as_shape": r["as_shape"],
                "bs_shape": r["bs_shape"],
                "a_stride": r["a_stride"],
                "b_stride": r["b_stride"],
                "as_stride": r["as_stride"],
                "bs_stride": r["bs_stride"],
                "a_dtype": r["a_dtype"],
                "b_dtype": r["b_dtype"],
                "as_dtype": r["as_dtype"],
                "bs_dtype": r["bs_dtype"],
                "output_dtype": r["output_dtype"],
                "block_size": r["block_size"],
                "regime": regime,
                "seed": 1701 + len(cases),
                "call_count": call_count,
                "captured_records": len(rows),
                "counts_by_rank": dict(sorted(by_rank.items())),
                "baseline_latency_ms": latency,
                "weight": call_count * latency if latency else float(call_count),
                "weight_source": (
                    "call_count_x_baseline_latency" if latency else "call_count"
                ),
            }
        )
    cases.sort(key=lambda c: (c["regime"], c["M"], c["N"], c["K"]))
    if not cases:
        raise SystemExit("capture contained no supported 2-D block-FP8 GEMM calls")

    baseline = Path(args.baseline_dir).resolve()
    baseline.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    cases_doc = {
        "schema": "geak.w8a8_blockscale_cases.v1",
        "source": manifest,
        "cases": cases,
    }
    (baseline / "cases.json").write_text(json.dumps(cases_doc, indent=2) + "\n")

    total_weight = sum(c["weight"] for c in cases) or 1.0
    workload_cases = []
    for c in cases:
        wc = {
            "sig": c["sig"],
            "M": c["M"],
            "N": c["N"],
            "K": c["K"],
            "regime": c["regime"],
            "seed": c["seed"],
            "dims": [c["a_shape"], c["b_shape"], c["as_shape"], c["bs_shape"]],
            "dtypes": [c["a_dtype"], c["b_dtype"], c["as_dtype"], c["bs_dtype"]],
            "strides": [c["a_stride"], c["b_stride"], c["as_stride"], c["bs_stride"]],
            "quant": {
                "scheme": "w8a8_fp8_block_scale",
                "block_size": c["block_size"],
                "output_dtype": c["output_dtype"],
                "weight_is_n_by_k": True,
                "scale_accumulation_dtype": "float32",
            },
            "count": c["call_count"],
            "baseline_latency_ms": c["baseline_latency_ms"],
            "weight": c["weight"],
            "weight_norm": c["weight"] / total_weight,
            "weight_source": c["weight_source"],
        }
        workload_cases.append(wc)
    workload = {
        "schema": "workload-v1",
        "target": "vllm TritonFp8BlockScaledMMKernel.apply_block_scaled_mm",
        "model": manifest["model"],
        "revision": manifest["revision"],
        "hardware": manifest.get("hardware", {}),
        "cases": workload_cases,
    }
    workload_path = baseline / "workload.json"
    workload_path.write_text(json.dumps(workload, indent=2) + "\n")

    root = Path(__file__).resolve().parents[2]
    op_shapes = [
        {k: c[k] for k in ("sig", "M", "N", "K", "block_size", "regime", "seed")}
        for c in cases
    ]
    bakeoff = {
        "kernel_path": str(baseline),
        "workflow_dir": str(root / "kernel_workflow"),
        "mode": "bakeoff",
        "backends": ["flydsl"],
        "gpu_ids": args.gpu_ids,
        "budget": args.budget,
        "apply_to_original": "false",
        "workload_spec_path": str(workload_path),
        "op_spec": {
            "op_kind": "gemm",
            "dtype": "fp8_e4m3fn",
            "output_dtype": "bfloat16",
            "block_size": [128, 128],
            "bias": False,
            "transpose_b": True,
            "regime": "both",
            "input_dtypes": ["float8_e4m3fn", "float8_e4m3fn", "float32", "float32"],
            "scale_dtype": "float32",
            "accumulation_dtype": "float32",
            "correctness": {"rtol": 0.01, "atol": 0.01, "random_draws": 5},
            "hardware": manifest.get("hardware", {}),
            "shapes": op_shapes,
            "layout_contract": (
                "A and activation scales are row-major. B is logical [N,K] but its row stride is "
                "captured independently and is padded to the checkpoint's physical K; the candidate "
                "must honor all workload case strides. B scales are row-major "
                "[ceil(N/128),ceil(K/128)]."
            ),
            "hardware_contract": (
                "Target gfx1201 RDNA4 wave32. The Radeon AI PRO R9700 has 64 physical CUs paired "
                "as 32 WGP scheduling/resource units; PyTorch multi_processor_count reports the 32 "
                "WGP-like units, not physical CUs. Size persistent/fill grids from WGP count and use "
                "WMMA, never CDNA MFMA."
            ),
            "math_contract": (
                "For each K block, accumulate FP32 dot(A_fp8, B_fp8) multiplied by the "
                "per-token-block A scale and per-[128,128] weight-block B scale; cast once "
                "to the captured output dtype. gfx1201 must use wave32 WMMA plus software "
                "FP32 block scaling; no CDNA MFMA, gfx1250 WMMA_SCALE, or AITER."
            ),
        },
        "task": (
            "Port the frozen vLLM Triton W8A8 FP8 block-scale GEMM to standalone FlyDSL main "
            "for gfx1201. Preserve exact captured layouts/dtypes and optimize the weighted "
            "Qwen3.8-27B-FP8 decode+prefill shape set."
        ),
    }
    Path(args.bakeoff_json).write_text(json.dumps(bakeoff, indent=2) + "\n")
    print(f"captured_records={len(records)} unique_cases={len(cases)}")
    print(f"cases={baseline / 'cases.json'}")
    print(f"workload={workload_path}")
    print(f"bakeoff={Path(args.bakeoff_json).resolve()}")


if __name__ == "__main__":
    main()
