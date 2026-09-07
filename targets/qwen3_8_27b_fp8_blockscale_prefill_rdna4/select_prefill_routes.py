#!/usr/bin/env python3
"""Select a small, explicit prefill routing policy from raw benchmark receipts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BENCH = HERE / "capture/prefill_variant_bench.json"
DEFAULT_POLICY = HERE / "prefill_selection_policy.json"
DEFAULT_OUTPUT = HERE / "capture/prefill_routes.json"
DEFAULT_PYTHON = HERE / "capture/generated_prefill_routes.py"
DEFAULT_MARKDOWN = HERE / "capture/PREFILL_ROUTE_SELECTION.md"


def _bucket_index(value: int, upper_bounds: list[int | None]) -> int:
    for index, upper in enumerate(upper_bounds):
        if upper is None or value <= upper:
            return index
    raise ValueError(f"no bucket for {value}")


def _bucket_key(row: dict, policy: dict) -> tuple[int, int, int]:
    return (
        _bucket_index(int(row["M"]), policy["m_bucket_upper_bounds"]),
        _bucket_index(int(row["N"]), policy["n_bucket_upper_bounds"]),
        _bucket_index(int(row["K"]), policy["k_bucket_upper_bounds"]),
    )


def _bounds(index: int, upper_bounds: list[int | None]) -> tuple[int, int | None]:
    lower = 1 if index == 0 else int(upper_bounds[index - 1]) + 1
    upper = upper_bounds[index]
    return lower, int(upper) if upper is not None else None


def _ratio_of_sums(speedups: list[float], weights: list[float] | None = None) -> float:
    if not speedups:
        return 1.0
    weights = weights or [1.0] * len(speedups)
    total = sum(weights)
    return total / sum(weight / speedup for weight, speedup in zip(weights, speedups))


def _geomean(values: list[float]) -> float:
    return (
        math.exp(sum(math.log(value) for value in values) / len(values))
        if values
        else 1.0
    )


def _variant_bucket_result(rows: list[dict], variant: str, policy: dict) -> dict:
    speedups = []
    conservative = []
    failures = []
    for row in rows:
        baseline = row["results"].get("baseline", {})
        result = row["results"].get(variant, {})
        if not result.get("correct") or not result.get("independent_output"):
            failures.append(f"{row['sig']}: correctness/output independence")
            continue
        if "speedup" not in result or "conservative_speedup" not in result:
            failures.append(f"{row['sig']}: missing timing")
            continue
        if float(baseline.get("spread", math.inf)) > float(policy["max_pass_spread"]):
            failures.append(f"{row['sig']}: baseline spread")
        if float(result.get("spread", math.inf)) > float(policy["max_pass_spread"]):
            failures.append(f"{row['sig']}: candidate spread")
        speedups.append(float(result["speedup"]))
        conservative.append(float(result["conservative_speedup"]))

    enough = len(rows) >= int(policy["min_samples_per_bucket"])
    bucket_speedup = _ratio_of_sums(speedups) if len(speedups) == len(rows) else 0.0
    minimum = (
        min(conservative) if conservative and len(conservative) == len(rows) else 0.0
    )
    eligible = (
        enough
        and not failures
        and minimum >= float(policy["per_case_regression_floor"])
        and bucket_speedup >= float(policy["min_bucket_speedup"])
    )
    return {
        "variant": variant,
        "eligible": eligible,
        "num_cases": len(rows),
        "bucket_speedup": bucket_speedup,
        "geomean_speedup": _geomean(speedups),
        "minimum_conservative_speedup": minimum,
        "failures": failures,
    }


def _choose_bucket_variant(
    eligible: dict[str, dict],
    allowed: set[str],
    policy: dict,
) -> str:
    candidates = [
        result
        for name, result in eligible.items()
        if name in allowed and result["eligible"]
    ]
    if not candidates:
        return policy["fallback"]
    best_score = max(result["bucket_speedup"] for result in candidates)
    near = [
        result
        for result in candidates
        if best_score - result["bucket_speedup"] <= float(policy["variant_tie_margin"])
    ]
    rank = {name: index for index, name in enumerate(policy["variant_preference"])}
    return min(near, key=lambda result: rank.get(result["variant"], 10_000))["variant"]


def _evaluate_policy(
    rows: list[dict],
    grouped: dict[tuple[int, int, int], list[dict]],
    eligibility: dict[tuple[int, int, int], dict[str, dict]],
    allowed: set[str],
    policy: dict,
) -> dict:
    routes = {
        key: _choose_bucket_variant(eligibility[key], allowed, policy)
        for key in grouped
    }
    speedups = []
    production_speedups = []
    production_weights = []
    for row in rows:
        variant = routes[_bucket_key(row, policy)]
        speedup = (
            1.0
            if variant == policy["fallback"]
            else float(row["results"][variant]["speedup"])
        )
        speedups.append(speedup)
        if float(row.get("production_weight", 0.0)) > 0:
            production_speedups.append(speedup)
            production_weights.append(float(row["production_weight"]))
    used = sorted(
        {variant for variant in routes.values() if variant != policy["fallback"]}
    )
    return {
        "allowed_variants": sorted(allowed),
        "used_variants": used,
        "generalization_speedup": _ratio_of_sums(speedups),
        "production_speedup": (
            _ratio_of_sums(production_speedups, production_weights)
            if production_speedups
            else 1.0
        ),
        "routes": routes,
    }


def _select_policy(candidates: list[dict], policy: dict) -> dict:
    best_score = max(item["generalization_speedup"] for item in candidates)
    near = [
        item
        for item in candidates
        if best_score - item["generalization_speedup"]
        <= float(policy["policy_tie_margin"])
    ]
    rank = {name: index for index, name in enumerate(policy["variant_preference"])}

    def key(item: dict) -> tuple:
        preference = tuple(
            sorted(rank.get(name, 10_000) for name in item["used_variants"])
        )
        return len(item["used_variants"]), preference, -item["generalization_speedup"]

    return min(near, key=key)


def _merge_routes(selected: dict, policy: dict) -> list[dict]:
    raw = []
    for (m_index, n_index, k_index), variant in selected["routes"].items():
        if variant == policy["fallback"]:
            continue
        m_min, m_max = _bounds(m_index, policy["m_bucket_upper_bounds"])
        n_min, n_max = _bounds(n_index, policy["n_bucket_upper_bounds"])
        k_min, k_max = _bounds(k_index, policy["k_bucket_upper_bounds"])
        raw.append(
            {
                "m_index": m_index,
                "m_min": m_min,
                "m_max": m_max,
                "n_min": n_min,
                "n_max": n_max,
                "k_min": k_min,
                "k_max": k_max,
                "variant": variant,
            }
        )

    merged = []
    for group_key, group_iter in itertools.groupby(
        sorted(
            raw,
            key=lambda route: (
                route["n_min"],
                route["k_min"],
                route["variant"],
                route["m_index"],
            ),
        ),
        key=lambda route: (
            route["n_min"],
            route["n_max"],
            route["k_min"],
            route["k_max"],
            route["variant"],
        ),
    ):
        group = list(group_iter)
        current = None
        for route in group:
            if current is not None and route["m_index"] == current["m_index_end"] + 1:
                current["m_max"] = route["m_max"]
                current["m_index_end"] = route["m_index"]
            else:
                if current is not None:
                    merged.append(current)
                current = {**route, "m_index_end": route["m_index"]}
        if current is not None:
            merged.append(current)
    for route in merged:
        route.pop("m_index")
        route.pop("m_index_end")
    return sorted(
        merged,
        key=lambda route: (
            route["m_min"],
            route["n_min"],
            route["k_min"],
            route["variant"],
        ),
    )


def _condition(name: str, low: int, high: int | None) -> str:
    if high is None:
        return f"{name} >= {low}"
    if low == 1:
        return f"{name} <= {high}"
    return f"{low} <= {name} <= {high}"


def _write_python(path: Path, routes: list[dict], provenance: str) -> None:
    lines = [
        '"""Generated coarse RDNA4 prefill routing policy; fallback is always safe."""',
        "",
        f'BENCHMARK_SHA256 = "{provenance}"',
        "",
        "def select_prefill_variant(M: int, N: int, K: int) -> str:",
    ]
    if not routes:
        lines.append('    return "baseline"')
    else:
        for route in routes:
            conditions = [
                _condition("M", route["m_min"], route["m_max"]),
                _condition("N", route["n_min"], route["n_max"]),
                _condition("K", route["k_min"], route["k_max"]),
            ]
            lines.append(f"    if {' and '.join(conditions)}:")
            lines.append(f'        return "{route["variant"]}"')
        lines.append('    return "baseline"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Deterministic prefill route selection",
        "",
        f"- Generalization score: **{result['generalization_speedup']:.6f}x**",
        f"- Captured-production diagnostic: **{result['production_speedup']:.6f}x**",
        f"- Custom variants retained: **{', '.join(result['used_variants']) or 'none'}**",
        f"- Fallback: **{result['fallback']}**",
        "",
        "| M range | N range | K range | Variant |",
        "|---|---|---|---|",
    ]
    for route in result["routes"]:
        show = lambda low, high: f"{low}+" if high is None else f"{low}–{high}"
        lines.append(
            f"| {show(route['m_min'], route['m_max'])} | "
            f"{show(route['n_min'], route['n_max'])} | "
            f"{show(route['k_min'], route['k_max'])} | {route['variant']} |"
        )
    if not result["routes"]:
        lines.append("| all | all | all | baseline |")
    lines.extend(
        [
            "",
            "Unlisted buckets use the vLLM baseline. A route appears only when every sampled case in its",
            "coarse bucket passed correctness, stability, minimum-speedup, and regression-floor gates.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-output", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    bench_bytes = args.bench.read_bytes()
    policy_bytes = args.policy.read_bytes()
    bench = json.loads(bench_bytes)
    policy = json.loads(policy_bytes)
    if bench.get("status") != "complete":
        raise SystemExit(
            "benchmark receipt is incomplete; resume it before selecting routes"
        )
    rows = bench["rows"]
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(_bucket_key(row, policy), []).append(row)

    eligibility = {
        key: {
            variant: _variant_bucket_result(group_rows, variant, policy)
            for variant in policy["candidate_variants"]
        }
        for key, group_rows in grouped.items()
    }
    all_variants = list(policy["candidate_variants"])
    candidate_policies = []
    for count in range(int(policy["max_custom_variants"]) + 1):
        for subset in itertools.combinations(all_variants, count):
            candidate_policies.append(
                _evaluate_policy(rows, grouped, eligibility, set(subset), policy)
            )
    selected = _select_policy(candidate_policies, policy)
    merged_routes = _merge_routes(selected, policy)
    result = {
        "schema": "geak.prefill_routes.v1",
        "benchmark": str(args.bench.resolve()),
        "benchmark_sha256": hashlib.sha256(bench_bytes).hexdigest(),
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "objective": policy["objective"],
        "generalization_speedup": selected["generalization_speedup"],
        "production_speedup": selected["production_speedup"],
        "used_variants": selected["used_variants"],
        "fallback": policy["fallback"],
        "routes": merged_routes,
        "bucket_eligibility": {
            f"m{key[0]}_n{key[1]}_k{key[2]}": value
            for key, value in sorted(eligibility.items())
        },
        "candidate_policies": [
            {key: value for key, value in candidate.items() if key != "routes"}
            for candidate in candidate_policies
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _write_python(args.python_output, merged_routes, result["benchmark_sha256"])
    _write_markdown(args.markdown_output, result)
    print(
        f"selected {result['used_variants'] or ['baseline']} with "
        f"generalization={result['generalization_speedup']:.6f}x, "
        f"production={result['production_speedup']:.6f}x"
    )
    print(f"wrote {args.output}, {args.python_output}, and {args.markdown_output}")


if __name__ == "__main__":
    main()
