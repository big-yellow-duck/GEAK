#!/usr/bin/env python3
"""Build the FlyDSL prefill campaign's scored and generalization cases."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEAK_ROOT = HERE.parents[1]
DEFAULT_CAPTURED = GEAK_ROOT / "targets/qwen3_8_27b_fp8_blockscale/baseline/cases.json"

HARDWARE = {
    "gfx": "gfx1201",
    "arch_class": "rdna4",
    "wave_size": 32,
    "physical_cu_count": 64,
    "wgp_count": 32,
}

WEIGHT_FAMILIES = (
    (5120, 3072),
    (5120, 8704),
    (7168, 5120),
    (8192, 5120),
    (17408, 5120),
)

# One shape for each existing M<=64 FlyDSL micro-route and important boundary.
FLYDSL_SEED_ROUTE_GUARDS = (
    (4, 5120, 3072),
    (4, 7168, 5120),
    (16, 5120, 8704),
    (17, 5120, 3072),
    (32, 17408, 5120),
    (33, 17408, 5120),
    (33, 5120, 8704),
    (48, 17408, 5120),
    (64, 5120, 3072),
)

# Boundaries previously shown to be discontinuous for the Triton campaign.
PREFILL_BOUNDARY_GUARDS = (
    (32, 5120, 8704),
    (33, 5120, 8704),
    (63, 17408, 5120),
    (64, 17408, 5120),
    (80, 5120, 8704),
    (127, 17408, 5120),
    (128, 5120, 8704),
    (129, 5120, 8704),
    (224, 8192, 5120),
    (225, 7168, 5120),
    (256, 8192, 5120),
    (257, 8192, 5120),
    (512, 5120, 8704),
    (768, 5120, 8704),
    (783, 5120, 8704),
    (785, 17408, 5120),
    (1024, 17408, 5120),
)

# Full production-family cross-sections: first unsupported M, a promising
# narrow-prefill bucket, and a mid-prefill bucket.
CROSS_SECTION_M = (65, 96, 240)

GENERALIZATION_GUARDS = (
    (192, 5120, 8704),
    (384, 17408, 5120),
    (640, 8192, 5120),
    (832, 7168, 5120),
)

# Non-Qwen N/K families prevent exact production-shape routing. Odd padded B
# row strides explicitly preserve the actual stride contract.
PORTABILITY_FAMILIES = (
    (128, 128, 131),
    (1024, 2176, 2307),
)
PORTABILITY_M = (65, 127, 128, 129, 255, 256, 257, 511, 512, 1024)
DECODE_K_GUARDS = (128, 384, 640, 2176)


def _zero_weight_case(template: dict, *, m: int, role: str, seed: int) -> dict:
    case = copy.deepcopy(template)
    n, k = int(case["N"]), int(case["K"])
    k_blocks = k // 128
    case.update(
        {
            "sig": f"{role}_m{m}_n{n}_k{k}_bn128_bk128",
            "M": m,
            "a_shape": [m, k],
            "as_shape": [m, k_blocks],
            "a_stride": [k, 1],
            "as_stride": [k_blocks, 1],
            "regime": "prefill",
            "seed": seed,
            "count": 0,
            "call_count": 0,
            "baseline_latency_ms": 0.0,
            "weight": 0.0,
            "weight_source": "zero_weight_generalization_gate",
            "coverage_role": role,
            "route_expectation": (
                "generalize_to_flydsl" if m > 64 else "preserve_flydsl_seed"
            ),
            "synthetic": True,
        }
    )
    for key in ("captured_records", "counts_by_rank"):
        case.pop(key, None)
    return case


def _portable_case(*, m: int, n: int, k: int, stride_b: int, seed: int) -> dict:
    k_blocks = k // 128
    return {
        "sig": f"portable_prefill_m{m}_n{n}_k{k}_bn128_bk128",
        "M": m,
        "N": n,
        "K": k,
        "a_shape": [m, k],
        "b_shape": [n, k],
        "as_shape": [m, k_blocks],
        "bs_shape": [n // 128, k_blocks],
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
        "regime": "prefill",
        "seed": seed,
        "count": 0,
        "call_count": 0,
        "baseline_latency_ms": 0.0,
        "weight": 0.0,
        "weight_source": "zero_weight_portability_gate",
        "coverage_role": "portable_prefill_generalization",
        "route_expectation": "generalize_to_flydsl",
        "synthetic": True,
    }


def _decode_k_case(template: dict, *, m: int, k: int, seed: int) -> dict:
    case = copy.deepcopy(template)
    n = int(case["N"])
    k_blocks = k // 128
    case.update(
        {
            "sig": f"decode_k_guard_m{m}_n{n}_k{k}_bn128_bk128",
            "M": m,
            "K": k,
            "a_shape": [m, k],
            "b_shape": [n, k],
            "as_shape": [m, k_blocks],
            "bs_shape": [n // 128, k_blocks],
            "a_stride": [k, 1],
            "b_stride": [k + 3, 1],
            "as_stride": [k_blocks, 1],
            "bs_stride": [k_blocks, 1],
            "regime": "decode",
            "seed": seed,
            "count": 0,
            "call_count": 0,
            "baseline_latency_ms": 0.0,
            "weight": 0.0,
            "weight_source": "zero_weight_decode_regression_gate",
            "coverage_role": "flydsl_decode_k_regression",
            "route_expectation": "preserve_flydsl_seed",
            "synthetic": True,
        }
    )
    for key in ("captured_records", "counts_by_rank"):
        case.pop(key, None)
    return case


def _workload_case(case: dict, total_weight: float) -> dict:
    weight = float(case["weight"])
    return {
        "sig": case["sig"],
        "M": case["M"],
        "N": case["N"],
        "K": case["K"],
        "regime": case["regime"],
        "coverage_role": case["coverage_role"],
        "route_expectation": case["route_expectation"],
        "seed": case["seed"],
        "dims": [case["a_shape"], case["b_shape"], case["as_shape"], case["bs_shape"]],
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
        "count": int(case["call_count"]),
        "baseline_latency_ms": float(case["baseline_latency_ms"]),
        "weight": weight,
        "weight_norm": weight / total_weight if total_weight else 0.0,
        "weight_source": case["weight_source"],
    }


def build(captured_path: Path) -> tuple[dict, dict]:
    captured = json.loads(captured_path.read_text())["cases"]
    templates = {(int(case["N"]), int(case["K"])): case for case in captured}
    cases = []
    used_dims = set()

    def add(case: dict) -> None:
        dims = (int(case["M"]), int(case["N"]), int(case["K"]))
        if dims in used_dims:
            return
        used_dims.add(dims)
        cases.append(case)

    for original in captured:
        case = copy.deepcopy(original)
        if case["regime"] == "prefill":
            case["coverage_role"] = "scored_captured_prefill"
            case["weight_source"] = "captured_call_count_x_vllm_baseline_latency"
            case["route_expectation"] = "generalize_to_flydsl"
        else:
            case["weight"] = 0.0
            case["weight_source"] = "zero_weight_flydsl_decode_regression_gate"
            case["coverage_role"] = "captured_flydsl_decode_regression"
            case["route_expectation"] = "preserve_flydsl_seed"
        add(case)

    seed = 1_200_000
    for m, n, k in FLYDSL_SEED_ROUTE_GUARDS:
        add(
            _zero_weight_case(
                templates[(n, k)], m=m, role="flydsl_seed_route_guard", seed=seed
            )
        )
        seed += 1

    requested = set(PREFILL_BOUNDARY_GUARDS)
    requested.update(GENERALIZATION_GUARDS)
    for m in CROSS_SECTION_M:
        requested.update((m, n, k) for n, k in WEIGHT_FAMILIES)
    for m, n, k in sorted(requested):
        add(
            _zero_weight_case(
                templates[(n, k)], m=m, role="heldout_prefill_generalization", seed=seed
            )
        )
        seed += 1

    for m in PORTABILITY_M:
        for n, k, stride_b in PORTABILITY_FAMILIES:
            add(_portable_case(m=m, n=n, k=k, stride_b=stride_b, seed=seed))
            seed += 1

    decode_template = templates[(5120, 3072)]
    for m in (1, 2):
        for k in DECODE_K_GUARDS:
            add(_decode_k_case(decode_template, m=m, k=k, seed=seed))
            seed += 1

    total_weight = sum(float(case["weight"]) for case in cases)
    cases_doc = {
        "schema": "geak.w8a8_blockscale_cases.v1",
        "source": {
            "captured_cases": str(captured_path),
            "model": "Qwen/Qwen3.8-27B-FP8",
            "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "hardware": HARDWARE,
            "scoring": "captured prefill weighted; FlyDSL seed, decode, boundary, and portability cases are hard gates",
        },
        "cases": cases,
    }
    workload_doc = {
        "schema": "workload-v1",
        "target": "Qwen3.8-27B FP8 blockscale FlyDSL broad-prefill generalization on RDNA4",
        "model": cases_doc["source"]["model"],
        "revision": cases_doc["source"]["revision"],
        "hardware": HARDWARE,
        "metric": "captured_prefill_counted_ratio_of_sums_with_broad_flydsl_hard_gates",
        "seed_weighted_total_ms": total_weight,
        "cases": [_workload_case(case, total_weight) for case in cases],
    }
    return cases_doc, workload_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured-cases", type=Path, default=DEFAULT_CAPTURED)
    parser.add_argument("--output-dir", type=Path, default=HERE / "baseline")
    args = parser.parse_args()

    cases_doc, workload_doc = build(args.captured_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cases.json").write_text(json.dumps(cases_doc, indent=2) + "\n")
    (args.output_dir / "workload.json").write_text(
        json.dumps(workload_doc, indent=2) + "\n"
    )
    roles = {}
    for case in cases_doc["cases"]:
        role = case["coverage_role"]
        roles[role] = roles.get(role, 0) + 1
    print(
        f"wrote {len(cases_doc['cases'])} cases, weighted_ms={workload_doc['seed_weighted_total_ms']:.6f}, "
        f"roles={roles}"
    )


if __name__ == "__main__":
    main()
