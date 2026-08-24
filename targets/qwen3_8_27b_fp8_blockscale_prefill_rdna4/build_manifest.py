#!/usr/bin/env python3
"""Build the scored prefill manifest plus zero-weight regression sentinels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEAK_ROOT = HERE.parents[1]
DEFAULT_CAPTURED = (
    GEAK_ROOT / "targets/qwen3_8_27b_fp8_blockscale/baseline/cases.json"
)
DEFAULT_WINNER = (
    GEAK_ROOT
    / "exp/bakeoff_baseline_20260822_102440/bakeoff/triton"
    / "team_task_20260822_103835_290494_26777/task/round_4"
    / "engineer_1/worker_result.json"
)

WEIGHT_FAMILIES = (
    (5120, 3072),
    (5120, 8704),
    (7168, 5120),
    (8192, 5120),
    (17408, 5120),
)

# For each observed BM32 discontinuity, retain the worst measured N/K family.
# These are correctness + regression sentinels, not synthetic production weight.
BOUNDARY_SENTINELS = (
    (32, 5120, 8704),
    (33, 5120, 8704),
    (63, 17408, 5120),
    (64, 17408, 5120),
    (80, 5120, 8704),
    (96, 5120, 8704),
    (128, 5120, 8704),
    (129, 5120, 8704),
    (224, 8192, 5120),
    (225, 7168, 5120),
    (240, 8192, 5120),
    (256, 8192, 5120),
    (257, 8192, 5120),
    (512, 5120, 8704),
    (768, 5120, 8704),
    (783, 5120, 8704),
    (785, 17408, 5120),
    (1024, 17408, 5120),
)

# Full N/K cross-sections prevent a route from being justified by one friendly
# family at the centers of the two promising tile-aligned bands.
CROSS_SECTION_M = (96, 240)

# These interior M values were not part of the captured workload or the dense
# crossover sweep. Rotate their N/K families so exact-M routing cannot masquerade
# as coverage of the spaces between known boundaries.
GENERALIZATION_SENTINELS = (
    (192, 5120, 8704),
    (384, 17408, 5120),
    (640, 8192, 5120),
    (832, 7168, 5120),
)

# Exercise the complete K%128 decode contract, including one idle split wave,
# uneven split-K partitions, and the Qwen3.8 TP4 projection missed by K%256.
DECODE_K_SENTINELS = (128, 384, 640, 2176)


def _workload_case(case: dict, total_weight: float) -> dict:
    weight = float(case["weight"])
    return {
        "sig": case["sig"],
        "M": case["M"],
        "N": case["N"],
        "K": case["K"],
        "regime": case["regime"],
        "coverage_role": case["coverage_role"],
        "seed": case["seed"],
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
        "count": int(case["call_count"]),
        "baseline_latency_ms": float(case["baseline_latency_ms"]),
        "weight": weight,
        "weight_norm": weight / total_weight if total_weight else 0.0,
        "weight_source": case["weight_source"],
    }


def _heldout_case(template: dict, m: int, family_index: int) -> dict:
    case = copy.deepcopy(template)
    n, k = int(case["N"]), int(case["K"])
    k_blocks = k // 128
    case.update(
        {
            "sig": f"holdout_m{m}_n{n}_k{k}_bn128_bk128",
            "M": m,
            "a_shape": [m, k],
            "as_shape": [m, k_blocks],
            "a_stride": [k, 1],
            "as_stride": [k_blocks, 1],
            "regime": "prefill",
            "seed": 900_000 + m * 10 + family_index,
            "call_count": 0,
            "baseline_latency_ms": 0.0,
            "weight": 0.0,
            "weight_source": "synthetic_zero_weight_regression_gate",
            "coverage_role": "heldout_prefill_regression",
            "synthetic": True,
        }
    )
    for key in ("captured_records", "counts_by_rank"):
        case.pop(key, None)
    return case


def _decode_heldout_case(template: dict, m: int, k: int) -> dict:
    case = copy.deepcopy(template)
    n = int(case["N"])
    k_blocks = k // 128
    case.update(
        {
            "sig": f"holdout_decode_m{m}_n{n}_k{k}_bn128_bk128",
            "M": m,
            "K": k,
            "a_shape": [m, k],
            "b_shape": [n, k],
            "as_shape": [m, k_blocks],
            "bs_shape": [n // 128, k_blocks],
            "a_stride": [k, 1],
            "b_stride": [k + 256, 1],
            "as_stride": [k_blocks, 1],
            "bs_stride": [k_blocks, 1],
            "regime": "decode",
            "seed": 950_000 + m * 10_000 + k,
            "call_count": 0,
            "baseline_latency_ms": 0.0,
            "weight": 0.0,
            "weight_source": "synthetic_zero_weight_regression_gate",
            "coverage_role": "heldout_decode_regression",
            "synthetic": True,
        }
    )
    for key in ("captured_records", "counts_by_rank"):
        case.pop(key, None)
    return case


def build(captured_path: Path, winner_path: Path) -> tuple[dict, dict]:
    captured_doc = json.loads(captured_path.read_text())
    winner = json.loads(winner_path.read_text())
    receipt = {row["name"]: row for row in winner["per_case"]}

    cases = []
    templates = {}
    for original in captured_doc["cases"]:
        case = copy.deepcopy(original)
        row = receipt[case["sig"]]
        case["baseline_latency_ms"] = float(row["optimized_ms"])
        if case["regime"] == "prefill":
            case["weight"] = int(case["call_count"]) * float(row["optimized_ms"])
            case["weight_source"] = "call_count_x_validated_seed_latency"
            case["coverage_role"] = "scored_captured_prefill"
        else:
            case["weight"] = 0.0
            case["weight_source"] = "zero_weight_decode_seed_guard"
            case["coverage_role"] = "decode_regression"
        cases.append(case)
        templates.setdefault((int(case["N"]), int(case["K"])), case)

    requested = set(BOUNDARY_SENTINELS)
    requested.update(GENERALIZATION_SENTINELS)
    for m in CROSS_SECTION_M:
        requested.update((m, n, k) for n, k in WEIGHT_FAMILIES)
    for m, n, k in sorted(requested):
        family_index = WEIGHT_FAMILIES.index((n, k))
        cases.append(_heldout_case(templates[(n, k)], m, family_index))

    decode_template = templates[(5120, 3072)]
    for m in (1, 2):
        for k in DECODE_K_SENTINELS:
            cases.append(_decode_heldout_case(decode_template, m, k))

    total_weight = sum(float(case["weight"]) for case in cases)
    cases_doc = {
        "schema": "geak.w8a8_blockscale_cases.v1",
        "source": {
            "captured_cases": str(captured_path),
            "validated_seed_receipt": str(winner_path),
            "model": "Qwen/Qwen3.8-27B-FP8",
            "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "hardware": {
                "gfx": "gfx1201",
                "arch_class": "rdna4",
                "wave_size": 32,
                "physical_cu_count": 64,
                "wgp_count": 32,
            },
            "scoring": "captured prefill only; decode and synthetic sentinels are zero-weight hard gates",
        },
        "cases": cases,
    }
    workload_doc = {
        "schema": "workload-v1",
        "target": "Qwen3.8-27B FP8 blockscale RDNA4 prefill specialization",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "hardware": cases_doc["source"]["hardware"],
        "metric": "captured_prefill_counted_ratio_of_sums",
        "seed_weighted_total_ms": total_weight,
        "cases": [_workload_case(case, total_weight) for case in cases],
    }
    return cases_doc, workload_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured-cases", type=Path, default=DEFAULT_CAPTURED)
    parser.add_argument("--winner-result", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--output-dir", type=Path, default=HERE / "baseline")
    args = parser.parse_args()

    cases_doc, workload_doc = build(args.captured_cases, args.winner_result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cases.json").write_text(json.dumps(cases_doc, indent=2) + "\n")
    (args.output_dir / "workload.json").write_text(
        json.dumps(workload_doc, indent=2) + "\n"
    )
    scored = sum(c["coverage_role"] == "scored_captured_prefill" for c in cases_doc["cases"])
    decode = sum(c["coverage_role"] == "decode_regression" for c in cases_doc["cases"])
    heldout_prefill = sum(
        c["coverage_role"] == "heldout_prefill_regression"
        for c in cases_doc["cases"]
    )
    heldout_decode = sum(
        c["coverage_role"] == "heldout_decode_regression"
        for c in cases_doc["cases"]
    )
    print(
        f"wrote {len(cases_doc['cases'])} cases: scored={scored}, "
        f"decode={decode}, heldout_prefill={heldout_prefill}, "
        f"heldout_decode={heldout_decode}, "
        f"seed_weighted_total_ms={workload_doc['seed_weighted_total_ms']:.6f}"
    )


if __name__ == "__main__":
    main()
