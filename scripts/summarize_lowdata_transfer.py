from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METHODS = (
    "scratch",
    "full_finetune",
    "frozen_spatial",
    "domain_adapter_only",
    "adapter_head",
)
FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.0)
DISPLAY = {
    "scratch": "Target-only scratch",
    "full_finetune": "Full fine-tuning",
    "frozen_spatial": "Frozen spatial encoder",
    "domain_adapter_only": "Domain adapter only",
    "adapter_head": "Adapter + head (ours)",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def manifest_rows(run_root: Path) -> list[dict[str, Any]]:
    paths = sorted((run_root / "manifests").glob("*.json"))
    if not paths:
        paths = sorted(run_root.glob("*/manifest.json"))
    by_cell: dict[tuple[str, float, int], dict[str, Any]] = {}
    for path in paths:
        manifest = read_json(path)
        if manifest.get("status") != "completed":
            continue
        method = str(manifest["method"])
        fraction = float(manifest["fraction"])
        seed = int(manifest["seed"])
        if seed != 123:
            raise ValueError(f"Non-formal seed in {path}: {seed}")
        key = (method, fraction, seed)
        if key in by_cell:
            raise ValueError(f"Duplicate completed cell: {key}")
        db = float(manifest["best_validation_nmse_db"])
        ratio = float(manifest["trainable_ratio_percent"])
        if not math.isfinite(db) or not math.isfinite(ratio):
            raise ValueError(f"Non-finite metric in {path}")
        by_cell[key] = {
            "method": method,
            "fraction": fraction,
            "seed": seed,
            "best_val_nmse_db": db,
            "trainable_ratio": ratio,
            "adaptation_minutes": manifest.get("adaptation_minutes"),
            "reused_or_new": manifest.get("reused_or_new", "NEW"),
        }
    rows = list(by_cell.values())
    for method in METHODS:
        ratios = {
            round(float(row["trainable_ratio"]), 12)
            for row in rows
            if row["method"] == method
        }
        if len(ratios) > 1:
            raise ValueError(f"Trainable ratio changed across fractions for {method}.")
    return sorted(rows, key=lambda row: (METHODS.index(row["method"]), row["fraction"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["method"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["method"], row["fraction"]): row for row in rows}
    table = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        ratio = method_rows[0]["trainable_ratio"] if method_rows else None
        timing = lookup.get((method, 0.05), {}).get("adaptation_minutes")
        table.append(
            {
                "method": DISPLAY[method],
                "method_id": method,
                "trainable_percent": ratio,
                "adaptation_time_at_5_percent_minutes": timing,
                "nmse_db": {
                    f"{round(100 * fraction)}%": lookup.get((method, fraction), {}).get(
                        "best_val_nmse_db"
                    )
                    for fraction in FRACTIONS
                },
            }
        )
    return table


def render_markdown(table: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Trainable (%) | Adapt. time at 5% (min) | 1% | 5% | 10% | 20% | 100% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        ratio = row["trainable_percent"]
        timing = row["adaptation_time_at_5_percent_minutes"]
        values = row["nmse_db"]
        fields = [
            row["method"],
            "—" if ratio is None else f"{float(ratio):.2f}",
            "—" if timing is None else f"{float(timing):.2f}",
            *("—" if values[key] is None else f"{float(values[key]):.2f}" for key in ("1%", "5%", "10%", "20%", "100%")),
        ]
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend(
        [
            "",
            "All low-data transfer results are reported for seed 123.",
            "Values are validation-selected NMSE (dB); no standard deviation is computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(run_root: Path) -> dict[str, Any]:
    rows = manifest_rows(run_root)
    results = run_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "raw_runs.csv", rows)
    table = table_payload(rows)
    flat_rows = []
    for row in table:
        flat_rows.append(
            {
                "method": row["method"],
                "trainable_percent": row["trainable_percent"],
                "adaptation_time_at_5_percent_minutes": row[
                    "adaptation_time_at_5_percent_minutes"
                ],
                **row["nmse_db"],
            }
        )
    write_csv(results / "table2_seed123.csv", flat_rows)
    payload = {
        "seed": 123,
        "single_seed": True,
        "standard_deviation_reported": False,
        "completed_cells": len(rows),
        "theoretical_cells": 25,
        "table": table,
    }
    (results / "table2_seed123.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results / "table2_seed123.md").write_text(
        render_markdown(table), encoding="utf-8"
    )
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Summarize V1 seed123 low-data transfer runs.")
    root.add_argument("--run-root", required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    result = summarize(Path(args.run_root).expanduser().resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
