#!/usr/bin/env python3
"""Regression tests for deterministic coarse prefill route selection."""

from __future__ import annotations

import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTOR_PATH = HERE / "select_prefill_routes.py"
POLICY_PATH = HERE / "prefill_selection_policy.json"


def _load_selector():
    spec = importlib.util.spec_from_file_location(
        "prefill_route_selector", SELECTOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SELECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTOR = _load_selector()


def _result(speedup: float, *, spread: float = 0.01, correct: bool = True) -> dict:
    return {
        "correct": correct,
        "independent_output": True,
        "speedup": speedup,
        "conservative_speedup": speedup,
        "spread": spread,
    }


def _row(m: int, speedups: dict[str, float], *, weight: float = 0.0) -> dict:
    return {
        "sig": f"m{m}_n5120_k3072",
        "M": m,
        "N": 5120,
        "K": 3072,
        "production_weight": weight,
        "results": {
            "baseline": {"spread": 0.01},
            **{name: _result(speedup) for name, speedup in speedups.items()},
        },
    }


class RouteSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads(POLICY_PATH.read_text())

    def _select(self, rows: list[dict]) -> dict:
        grouped: dict[tuple[int, int, int], list[dict]] = {}
        for row in rows:
            key = SELECTOR._bucket_key(row, self.policy)
            grouped.setdefault(key, []).append(row)
        eligibility = {
            key: {
                variant: SELECTOR._variant_bucket_result(
                    group_rows, variant, self.policy
                )
                for variant in self.policy["candidate_variants"]
            }
            for key, group_rows in grouped.items()
        }
        candidates = []
        variants = self.policy["candidate_variants"]
        for count in range(self.policy["max_custom_variants"] + 1):
            for subset in itertools.combinations(variants, count):
                candidates.append(
                    SELECTOR._evaluate_policy(
                        rows, grouped, eligibility, set(subset), self.policy
                    )
                )
        return SELECTOR._select_policy(candidates, self.policy)

    def test_prefers_simple_bm64_plus_bm80_and_falls_back_on_regression(self) -> None:
        rows = [
            # BM32 is fractionally faster, but inside the 1% per-bucket tie margin;
            # the explicit preference keeps the general BM64 body.
            _row(64, {"bm32": 1.054, "bm64": 1.050, "bm80": 0.990}, weight=2),
            _row(96, {"bm32": 1.044, "bm64": 1.040, "bm80": 0.990}, weight=1),
            # BM80 earns a second body for the next whole coarse bucket.
            _row(192, {"bm32": 1.000, "bm64": 1.020, "bm80": 1.090}, weight=1),
            _row(240, {"bm32": 1.000, "bm64": 1.020, "bm80": 1.080}, weight=1),
            # One below-floor case rejects every custom variant for this bucket.
            _row(384, {"bm32": 0.970, "bm64": 0.970, "bm80": 0.970}),
            _row(400, {"bm32": 1.120, "bm64": 1.120, "bm80": 1.120}),
        ]
        selected = self._select(rows)
        routes = {
            SELECTOR._bucket_key(row, self.policy): variant
            for row, variant in (
                (row, selected["routes"][SELECTOR._bucket_key(row, self.policy)])
                for row in rows
            )
        }

        self.assertEqual(selected["used_variants"], ["bm64", "bm80"])
        self.assertEqual(routes[SELECTOR._bucket_key(rows[0], self.policy)], "bm64")
        self.assertEqual(routes[SELECTOR._bucket_key(rows[2], self.policy)], "bm80")
        self.assertEqual(routes[SELECTOR._bucket_key(rows[4], self.policy)], "baseline")

        merged = SELECTOR._merge_routes(selected, self.policy)
        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "routes.py"
            SELECTOR._write_python(generated, merged, "synthetic")
            namespace: dict = {}
            exec(compile(generated.read_text(), str(generated), "exec"), namespace)
            route = namespace["select_prefill_variant"]
            self.assertEqual(route(64, 5120, 3072), "bm64")
            self.assertEqual(route(192, 5120, 3072), "bm80")
            self.assertEqual(route(384, 5120, 3072), "baseline")
            self.assertEqual(route(64, 16384, 8192), "baseline")

    def test_noise_and_under_sampling_reject_a_bucket(self) -> None:
        rows = [
            _row(64, {"bm32": 1.20, "bm64": 1.20, "bm80": 1.20}),
            _row(96, {"bm32": 1.20, "bm64": 1.20, "bm80": 1.20}),
        ]
        rows[1]["results"]["bm64"]["spread"] = 0.20
        result = SELECTOR._variant_bucket_result(rows, "bm64", self.policy)
        self.assertFalse(result["eligible"])
        self.assertTrue(
            any("candidate spread" in failure for failure in result["failures"])
        )

        result = SELECTOR._variant_bucket_result(rows[:1], "bm32", self.policy)
        self.assertFalse(result["eligible"])


if __name__ == "__main__":
    unittest.main()
