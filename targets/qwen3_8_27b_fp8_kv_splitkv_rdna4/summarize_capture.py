#!/usr/bin/env python3
"""Summarize a paged-attention JSONL capture without retaining process IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(row["query"]["shape"]),
        int(row["max_seq_len"]),
        tuple(row["seq_lens"]["shape"]),
        tuple(row["block_table"]["shape"]),
    )


def _layout(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(row["key_cache"]["shape"]),
        tuple(row["key_cache"]["stride"]),
        row["key_cache"]["dtype"],
        tuple(row["value_cache"]["shape"]),
        tuple(row["value_cache"]["stride"]),
        row["value_cache"]["dtype"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.shape_log.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no records in {args.shape_log}")

    decode = [row for row in rows if int(row["max_query_len"]) == 1]
    signature_counts = Counter(_signature(row) for row in decode)
    layout_counts = Counter(_layout(row) for row in rows)
    scale_counts = Counter(
        (
            row["kv_cache_dtype"],
            row["k_scale_value"],
            row["v_scale_value"],
            bool(row["unit_kv_scale"]),
        )
        for row in rows
    )

    summary = {
        "schema": "geak.splitkv_paged_decode_capture_summary.v1",
        "record_count": len(rows),
        "worker_count": len({row["pid"] for row in rows}),
        "decode_record_count": len(decode),
        "full_attention_calls_per_decode_signature": min(signature_counts.values()),
        "decode_signatures": [
            {
                "record_count": count,
                "query_shape": list(signature[0]),
                "max_seq_len": signature[1],
                "seq_lens_shape": list(signature[2]),
                "block_table_shape": list(signature[3]),
            }
            for signature, count in sorted(
                signature_counts.items(), key=lambda item: (item[0][0][0], item[0][1])
            )
        ],
        "cache_layouts": [
            {
                "record_count": count,
                "key_shape": list(layout[0]),
                "key_stride": list(layout[1]),
                "key_dtype": layout[2],
                "value_shape": list(layout[3]),
                "value_stride": list(layout[4]),
                "value_dtype": layout[5],
            }
            for layout, count in layout_counts.items()
        ],
        "scale_contracts": [
            {
                "record_count": count,
                "kv_cache_dtype": scale[0],
                "k_scale": scale[1],
                "v_scale": scale[2],
                "unit_kv_scale_hint": scale[3],
            }
            for scale, count in scale_counts.items()
        ],
    }

    payload = json.dumps(summary, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
