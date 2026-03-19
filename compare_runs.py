from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_TSV = "./runs/autoresearch_5090/results.tsv"


def parse_float(value: str) -> float | None:
    if value in {"", "null", "None"}:
        return None
    return float(value)


def parse_int(value: str) -> int | None:
    if value in {"", "null", "None"}:
        return None
    return int(value)


def load_results_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parsed = dict(row)
            parsed["val_bpb"] = parse_float(row.get("val_bpb", ""))
            parsed["training_seconds"] = parse_float(row.get("training_seconds", ""))
            parsed["total_seconds"] = parse_float(row.get("total_seconds", ""))
            parsed["peak_vram_mb"] = parse_float(row.get("peak_vram_mb", ""))
            parsed["total_tokens_M"] = parse_float(row.get("total_tokens_M", ""))
            parsed["num_params_M"] = parse_float(row.get("num_params_M", ""))
            parsed["artifact_bytes"] = parse_int(row.get("artifact_bytes", ""))
            parsed["num_steps"] = parse_int(row.get("num_steps", ""))
            parsed["depth"] = parse_int(row.get("depth", ""))
            rows.append(parsed)
    return rows


def row_sort_key(row: dict[str, Any], sort_by: str) -> tuple[float, str]:
    value = row.get(sort_by)
    if value is None:
        return (float("inf"), str(row.get("run_id", "")))
    return (float(value), str(row.get("run_id", "")))


def bar(value: float | None, lo: float, hi: float, width: int = 20) -> str:
    if value is None:
        return "." * width
    if hi <= lo:
        return "#" * width
    ratio = (hi - float(value)) / (hi - lo)
    filled = max(1, min(width, int(round(ratio * width))))
    return "#" * filled + "." * (width - filled)


def render_rows(rows: list[dict[str, Any]], sort_by: str, limit: int) -> str:
    if not rows:
        return "No runs found."
    ordered = sorted(rows, key=lambda row: row_sort_key(row, sort_by))
    shown = ordered[:limit]
    numeric_values = [float(row[sort_by]) for row in shown if row.get(sort_by) is not None]
    lo = min(numeric_values) if numeric_values else 0.0
    hi = max(numeric_values) if numeric_values else 1.0
    lines = [
        f"{'run_id':<28} {'val_bpb':>10} {'artifact_MB':>11} {'train_s':>10} {'steps':>7} {'depth':>7} chart",
        "-" * 100,
    ]
    for row in shown:
        artifact_mb = None if row.get("artifact_bytes") is None else float(row["artifact_bytes"]) / 1e6
        lines.append(
            "{run_id:<28} {val_bpb:>10} {artifact_mb:>11} {training_seconds:>10} {num_steps:>7} {depth:>7} {chart}".format(
                run_id=str(row.get("run_id", ""))[:28],
                val_bpb="null" if row.get("val_bpb") is None else f"{float(row['val_bpb']):.6f}",
                artifact_mb="null" if artifact_mb is None else f"{artifact_mb:.3f}",
                training_seconds="null" if row.get("training_seconds") is None else f"{float(row['training_seconds']):.2f}",
                num_steps="null" if row.get("num_steps") is None else str(int(row["num_steps"])),
                depth="null" if row.get("depth") is None else str(int(row["depth"])),
                chart=bar(row.get(sort_by), lo, hi),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a compact leaderboard view from results.tsv.")
    parser.add_argument("--results_tsv", type=str, default=DEFAULT_RESULTS_TSV)
    parser.add_argument("--sort_by", type=str, default="val_bpb", choices=("val_bpb", "artifact_bytes", "training_seconds"))
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    rows = load_results_rows(Path(args.results_tsv))
    print(render_rows(rows, sort_by=args.sort_by, limit=args.limit))


if __name__ == "__main__":
    main()
