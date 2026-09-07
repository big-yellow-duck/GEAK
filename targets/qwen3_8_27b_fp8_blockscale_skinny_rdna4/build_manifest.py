#!/usr/bin/env python3
"""Build the deterministic M=1..16 skinny-GEMM cases and workload."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"

HARDWARE = {
    "gfx": "gfx1201",
    "arch_class": "rdna4",
    "wave_size": 32,
    "physical_cu_count": 64,
    "wgp_count": 32,
}

# Post-NCCL_PROTO=Simple rank-0 trace. Calls are per steady decode step and
# latency is the kernel average across 31 captured M=4 steps.
QWEN_FAMILIES = (
    (5120, 3072, 3328, 64, 0.04433),
    (5120, 8704, 8960, 64, 0.10782),
    (7168, 5120, 5376, 16, 0.06908),
    (8192, 5120, 5376, 48, 0.07562),
    (17408, 5120, 5376, 64, 0.16347),
)

PORTABILITY_FAMILIES = (
    (128, 128, 128),
    (1024, 2176, 2176),
)


def case_for(
    *,
    m: int,
    n: int,
    k: int,
    stride_b: int,
    seed: int,
    role: str,
    count: int = 0,
    latency_ms: float = 0.0,
) -> dict:
    k_blocks = k // 128
    n_blocks = n // 128
    weight = count * latency_ms
    return {
        "sig": f"skinny_m{m}_n{n}_k{k}_bn128_bk128",
        "M": m,
        "N": n,
        "K": k,
        "a_shape": [m, k],
        "b_shape": [n, k],
        "as_shape": [m, k_blocks],
        "bs_shape": [n_blocks, k_blocks],
        "a_stride": [k, 1],
        "b_stride": [stride_b, 1],
        "as_stride": [k_blocks, 1],
        "bs_stride": [k_blocks, 1],
        "a_dtype": "float8_e4m3fn",
        "b_dtype": "float8_e4m3fn",
        "as_dtype": "float32",
        "bs_dtype": "float32",
        "output_dtype": "bfloat16",
        "block_size": [128, 128],
        "regime": "decode",
        "seed": seed,
        "count": count,
        "call_count": count,
        "baseline_latency_ms": latency_ms,
        "weight": weight,
        "weight_source": (
            "post_simple_trace_call_count_x_latency"
            if weight
            else "zero_weight_native_coverage_gate"
        ),
        "coverage_role": role,
    }


def workload_case(case: dict, total_weight: float) -> dict:
    result = {
        key: case[key]
        for key in (
            "sig",
            "M",
            "N",
            "K",
            "regime",
            "coverage_role",
            "seed",
            "count",
            "baseline_latency_ms",
            "weight",
            "weight_source",
        )
    }
    result.update(
        {
            "dims": [
                case["a_shape"],
                case["b_shape"],
                case["as_shape"],
                case["bs_shape"],
            ],
            "dtypes": [
                case["a_dtype"],
                case["b_dtype"],
                case["as_dtype"],
                case["bs_dtype"],
            ],
            "strides": [
                case["a_stride"],
                case["b_stride"],
                case["as_stride"],
                case["bs_stride"],
            ],
            "quant": {
                "scheme": "w8a8_fp8_block_scale",
                "block_size": [128, 128],
                "output_dtype": "bfloat16",
                "weight_is_n_by_k": True,
                "scale_accumulation_dtype": "float32",
            },
            "weight_norm": case["weight"] / total_weight if total_weight else 0.0,
        }
    )
    return result


def main() -> None:
    cases = []
    seed = 4100
    for m in range(1, 17):
        for n, k, stride_b, m4_calls, m4_latency in QWEN_FAMILIES:
            cases.append(
                case_for(
                    m=m,
                    n=n,
                    k=k,
                    stride_b=stride_b,
                    seed=seed,
                    role="qwen_skinny",
                    count=m4_calls if m == 4 else 0,
                    latency_ms=m4_latency if m == 4 else 0.0,
                )
            )
            seed += 1
    for m in range(1, 17):
        for n, k, stride_b in PORTABILITY_FAMILIES:
            cases.append(
                case_for(
                    m=m,
                    n=n,
                    k=k,
                    stride_b=stride_b,
                    seed=seed,
                    role="portable_k128_sentinel",
                )
            )
            seed += 1

    assert len(cases) == 112
    assert sorted({c["M"] for c in cases}) == list(range(1, 17))
    total_weight = sum(float(c["weight"]) for c in cases)
    assert abs(total_weight - 24.93472) < 1e-6

    source = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "profile": "offline_tp2_collectives_simple_e2c7/rank0",
        "hardware": HARDWARE,
        "scoring": "production M=4 weighted; complete M=1..16 native coverage is a hard gate",
    }
    cases_doc = {
        "schema": "geak.w8a8_blockscale_cases.v1",
        "source": source,
        "cases": cases,
    }
    workload_doc = {
        "schema": "workload-v1",
        "target": "Qwen3.8-27B FP8 blockscale RDNA4 native M=1..16 skinny GEMM",
        "model": source["model"],
        "revision": source["revision"],
        "hardware": HARDWARE,
        "metric": "captured_m4_counted_ratio_of_sums_with_m1_16_native_hard_gates",
        "seed_weighted_total_ms": total_weight,
        "cases": [workload_case(case, total_weight) for case in cases],
    }
    BASELINE.mkdir(parents=True, exist_ok=True)
    (BASELINE / "cases.json").write_text(json.dumps(cases_doc, indent=2) + "\n")
    (BASELINE / "workload.json").write_text(json.dumps(workload_doc, indent=2) + "\n")
    print(f"wrote {len(cases)} cases; production M=4 weight={total_weight:.5f} ms/step")


if __name__ == "__main__":
    main()
