#!/usr/bin/env python3
"""Generate the focused M256 case manifest and GEAK workload."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"
PRIMARY_BASELINE_MS = 0.160477

SPECS = [
    # signature suffix, M, N, K, B row stride, seed, scored
    ("primary", 256, 8192, 5120, 5376, 1200035, True),
    ("m240_padded", 240, 8192, 5120, 5376, 2200001, False),
    ("m255_tail_padded", 255, 8192, 5120, 5376, 2200002, False),
    ("m257_tail_padded", 257, 8192, 5120, 5376, 2200003, False),
    ("m272_padded", 272, 8192, 5120, 5376, 2200004, False),
    ("m256_contiguous_b", 256, 8192, 5120, 5120, 2200005, False),
    ("m256_n5120_k8704_padded", 256, 5120, 8704, 8960, 2200006, False),
]


def make_case(spec: tuple[str, int, int, int, int, int, bool]) -> dict:
    suffix, m, n, k, b_stride, seed, scored = spec
    k_blocks = k // 128
    n_blocks = n // 128
    latency = PRIMARY_BASELINE_MS if scored else 0.0
    role = "scored_m256_primary" if scored else "zero_weight_generality_gate"
    return {
        "sig": f"m256_hip_{suffix}_m{m}_n{n}_k{k}_bn128_bk128",
        "M": m,
        "N": n,
        "K": k,
        "a_shape": [m, k],
        "b_shape": [n, k],
        "as_shape": [m, k_blocks],
        "bs_shape": [n_blocks, k_blocks],
        "a_stride": [k, 1],
        "b_stride": [b_stride, 1],
        "as_stride": [k_blocks, 1],
        "bs_stride": [k_blocks, 1],
        "a_dtype": "float8_e4m3fn",
        "b_dtype": "float8_e4m3fn",
        "as_dtype": "float32",
        "bs_dtype": "float32",
        "output_dtype": "bfloat16",
        "block_size": [128, 128],
        "regime": "prefill",
        "seed": seed,
        "call_count": 1 if scored else 0,
        "baseline_latency_ms": latency,
        "weight": latency,
        "weight_source": (
            "archived_geak_primary_latency" if scored else "zero_weight_hard_gate"
        ),
        "count": 1 if scored else 0,
        "coverage_role": role,
        "route_expectation": "native_hip_general_route",
        "synthetic": not scored,
    }


def workload_case(case: dict) -> dict:
    return {
        "sig": case["sig"],
        "M": case["M"],
        "N": case["N"],
        "K": case["K"],
        "regime": case["regime"],
        "coverage_role": case["coverage_role"],
        "route_expectation": case["route_expectation"],
        "seed": case["seed"],
        "dims": [case[f"{name}_shape"] for name in ("a", "b", "as", "bs")],
        "dtypes": [case[f"{name}_dtype"] for name in ("a", "b", "as", "bs")],
        "strides": [case[f"{name}_stride"] for name in ("a", "b", "as", "bs")],
        "quant": {
            "scheme": "w8a8_fp8_block_scale",
            "block_size": [128, 128],
            "output_dtype": "bfloat16",
            "weight_is_n_by_k": True,
            "scale_accumulation_dtype": "float32",
        },
        "count": case["count"],
        "baseline_latency_ms": case["baseline_latency_ms"],
        "weight": case["weight"],
        "weight_norm": 1.0 if case["weight"] else 0.0,
        "weight_source": case["weight_source"],
    }


def main() -> None:
    cases = [make_case(spec) for spec in SPECS]
    BASELINE.mkdir(parents=True, exist_ok=True)
    cases_doc = {
        "schema": "geak.w8a8_blockscale_cases.v1",
        "source": "focused M256 HIP-vs-Triton RDNA4 campaign",
        "cases": cases,
    }
    workload = {
        "schema": "workload-v1",
        "target": "Qwen3.8-27B FP8 blockscale M256 native HIP on RDNA4",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "focused-m256-hip-v1",
        "hardware": {
            "gfx": "gfx1201",
            "arch_class": "rdna4",
            "wave_size": 32,
            "physical_cu_count": 64,
            "wgp_count": 32,
        },
        "metric": "single_primary_ratio_with_zero_weight_native_hip_generality_gates",
        "seed_weighted_total_ms": PRIMARY_BASELINE_MS,
        "cases": [workload_case(case) for case in cases],
    }
    (BASELINE / "cases.json").write_text(json.dumps(cases_doc, indent=2) + "\n")
    (BASELINE / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    print(f"wrote {len(cases)} cases: 1 scored primary + {len(cases) - 1} hard gates")


if __name__ == "__main__":
    main()
