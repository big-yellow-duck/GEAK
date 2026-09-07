#!/usr/bin/env python3
"""Build the broad RDNA4 SplitKV fallback campaign manifest."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"


def case(
    name: str,
    *,
    query_dtype: str = "bfloat16",
    kv_dtype: str = "fp8",
    head_size: int = 256,
    gqa: int = 1,
    kv_heads: int = 1,
    page_size: int = 32,
    seq_lens: tuple[int, ...] = (1024,),
    splits: int = 8,
    scale: float | None = None,
    input_scale: float = 0.25,
    k_scale: float = 0.73,
    v_scale: float = 1.27,
    weight: float = 1.0,
    role: str = "scored_fallback",
    seed: int,
) -> dict:
    if scale is None:
        scale = head_size**-0.5
    if kv_dtype not in ("fp8", "fp8fnuz"):
        k_scale = v_scale = 1.0
    return {
        "name": name,
        "query_dtype": query_dtype,
        "kv_dtype": kv_dtype,
        "head_size": head_size,
        "num_query_heads": kv_heads * gqa,
        "num_kv_heads": kv_heads,
        "gqa_ratio": gqa,
        "page_size": page_size,
        "seq_lens": list(seq_lens),
        "splits": splits,
        "scale": scale,
        "input_scale": input_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "weight": weight,
        "count": 1 if weight else 0,
        "coverage_role": role,
        "seed": seed,
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    seed = 2026090400

    # Every Triton-supported GQA ratio is represented. GQA 6/7 are zero-weight
    # regression guards because the current FlyDSL seed already owns them.
    pages = (16, 32, 128, 544, 1056, 1568)
    lengths = (257, 1024, 2048, 4014, 4097, 8192)
    for gqa in range(1, 17):
        page = pages[(gqa - 1) % len(pages)]
        length = lengths[(gqa - 1) % len(lengths)]
        seed += 1
        seeded = gqa in (6, 7)
        cases.append(
            case(
                f"{'guard_seed' if seeded else 'score'}_fp8_bf16_d256_g{gqa}_p{page}_s{length}",
                gqa=gqa,
                kv_heads=1 if gqa <= 4 else 2,
                page_size=page,
                seq_lens=(length,),
                splits=(4, 8, 16)[gqa % 3],
                input_scale=0.4 if gqa in (1, 4, 5, 8) else 0.25,
                weight=0.0 if seeded else 1.0,
                role="seed_flydsl_guard" if seeded else "scored_gqa_ratio",
                seed=seed,
            )
        )

    # Cross the remaining query/cache dtype and head-size regimes at the GQA
    # corners. These all begin on Triton and are primary optimization cases.
    regimes = (
        ("bfloat16", "bfloat16", 256),
        ("float16", "fp8", 256),
        ("float16", "float16", 256),
        ("bfloat16", "fp8", 128),
        ("bfloat16", "bfloat16", 128),
        ("float16", "fp8", 128),
        ("float16", "float16", 128),
        ("bfloat16", "fp8fnuz", 256),
        ("float16", "fp8fnuz", 128),
    )
    for regime_index, (query_dtype, kv_dtype, head_size) in enumerate(regimes):
        for corner_index, gqa in enumerate((1, 4, 8, 16)):
            page = pages[(regime_index + corner_index) % len(pages)]
            length = (1024, 2048, 4097, 8192)[corner_index]
            seed += 1
            cases.append(
                case(
                    f"score_{kv_dtype}_{query_dtype}_d{head_size}_g{gqa}_p{page}_s{length}",
                    query_dtype=query_dtype,
                    kv_dtype=kv_dtype,
                    head_size=head_size,
                    gqa=gqa,
                    kv_heads=(1, 2, 2, 4)[corner_index],
                    page_size=page,
                    seq_lens=(length,),
                    splits=(4, 8, 16, 16)[corner_index],
                    input_scale=0.35,
                    role="scored_dtype_head",
                    seed=seed,
                )
            )

    # Ragged scheduling is a separate first-class surface, rather than an
    # accidental extrapolation from batch one.
    ragged_regimes = (
        ("bfloat16", "fp8", 256, 3, 5),
        ("bfloat16", "fp8", 256, 8, 12),
        ("bfloat16", "bfloat16", 256, 3, 8),
        ("float16", "fp8", 256, 3, 16),
        ("float16", "float16", 256, 8, 4),
        ("bfloat16", "fp8", 128, 3, 8),
        ("bfloat16", "bfloat16", 128, 8, 16),
        ("float16", "float16", 128, 3, 1),
    )
    for index, (query_dtype, kv_dtype, head_size, batch, gqa) in enumerate(
        ragged_regimes
    ):
        seed += 1
        seq_lens = tuple(
            8192 - row * (640 if batch == 3 else 384) for row in range(batch)
        )
        cases.append(
            case(
                f"score_ragged_b{batch}_{kv_dtype}_{query_dtype}_d{head_size}_g{gqa}",
                query_dtype=query_dtype,
                kv_dtype=kv_dtype,
                head_size=head_size,
                gqa=gqa,
                kv_heads=2,
                page_size=pages[index % len(pages)],
                seq_lens=seq_lens,
                splits=16,
                input_scale=0.3,
                role="scored_ragged_batch",
                seed=seed,
            )
        )

    # Boundary, arbitrary-scale, and padded-cache guards must never be traded
    # away for aggregate speed. They are deliberately zero weight.
    for index, (page, length, splits) in enumerate(
        (
            (16, 15, 2),
            (16, 16, 2),
            (16, 17, 2),
            (544, 543, 4),
            (544, 544, 4),
            (544, 545, 4),
        )
    ):
        seed += 1
        cases.append(
            case(
                f"guard_page_boundary_p{page}_s{length}",
                gqa=(1, 4, 5, 8, 12, 16)[index],
                kv_heads=2,
                page_size=page,
                seq_lens=(length,),
                splits=splits,
                scale=(0.031, 0.057, 0.071)[index % 3],
                input_scale=0.4,
                weight=0.0,
                role="boundary_accuracy_guard",
                seed=seed,
            )
        )

    return cases


def main() -> None:
    cases = build_cases()
    payload = {
        "schema": "geak.rdna4_splitkv_full_fallback_cases.v1",
        "notes": "Weighted cases cover the Triton SplitKV domain; zero-weight cases are hard gates.",
        "cases": cases,
    }
    (BASELINE / "cases.json").write_text(json.dumps(payload, indent=2) + "\n")
    workload = {
        "schema": "workload-v1",
        "target": "RDNA4 FlyDSL replacement of vLLM Triton SplitKV fallback",
        "metric": "broad_fallback_ratio_of_sums",
        "cases_path": str(BASELINE / "cases.json"),
        "notes": "Equal per-case weights across GQA, dtype/head, and ragged fallback surfaces; zero-weight seed and boundary guards.",
        "hardware": {
            "gfx": "gfx1201",
            "arch_class": "rdna4",
            "wave_size": 32,
            "physical_cu_count": 64,
            "wgp_count": 32,
            "geak_gpu_id": 1,
        },
    }
    (BASELINE / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    print(
        f"wrote {len(cases)} cases ({sum(bool(item['weight']) for item in cases)} scored)"
    )


if __name__ == "__main__":
    main()
