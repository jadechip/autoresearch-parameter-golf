from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PATH = "./runs/autoresearch_5090/index/latest.json"
DEFAULT_METRICS = ("train_loss", "matrix_lr", "val_bpb", "step_seconds")
ASCII_LEVELS = " .:-=+*#%@"
METRIC_FIELDS = {
    "train_loss": ("train", "train_loss"),
    "matrix_lr": ("train", "matrix_lr"),
    "embed_lr": ("train", "embed_lr"),
    "head_lr": ("train", "head_lr"),
    "scalar_lr": ("train", "scalar_lr"),
    "step_seconds": ("train", "step_seconds"),
    "val_loss": ("val", "val_loss"),
    "val_bpb": ("val", "val_bpb"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def resolve_run_paths(path_arg: str) -> tuple[Path, Path, Path, Path]:
    path = Path(path_arg)
    if path.is_dir():
        run_dir = path
        metrics_path = run_dir / "metrics.jsonl"
        return run_dir, metrics_path, run_dir / "results.json", run_dir / "crash.json"
    if path.suffix == ".jsonl":
        run_dir = path.parent
        return run_dir, path, run_dir / "results.json", run_dir / "crash.json"
    if path.suffix == ".json":
        payload = load_json(path)
        run_dir = Path(payload.get("output_dir", path.parent))
        metrics_path = Path(payload.get("metrics_path", run_dir / "metrics.jsonl"))
        results_path = Path(payload.get("results_path", run_dir / "results.json"))
        return run_dir, metrics_path, results_path, run_dir / "crash.json"
    raise ValueError(f"unsupported watch target: {path_arg}")


def parse_metric_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [name for name in names if name not in METRIC_FIELDS]
    if invalid:
        raise ValueError(f"unsupported metrics: {', '.join(invalid)}")
    return names


def last_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == event_name:
            return event
    return None


def metric_series(events: list[dict[str, Any]], metric_name: str, limit: int) -> list[float]:
    event_name, field_name = METRIC_FIELDS[metric_name]
    values: list[float] = []
    for event in events:
        if event.get("event") != event_name:
            continue
        value = event.get(field_name)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values[-limit:]


def sparkline(values: list[float], width: int) -> str:
    if width <= 0:
        return ""
    if not values:
        return "(no data)"
    if len(values) > width:
        step = len(values) / width
        sampled = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    else:
        sampled = values
    lo = min(sampled)
    hi = max(sampled)
    if hi <= lo:
        glyph = ASCII_LEVELS[len(ASCII_LEVELS) // 2]
        return glyph * max(1, len(sampled))
    chars = []
    scale = len(ASCII_LEVELS) - 1
    for value in sampled:
        idx = int(round((value - lo) / (hi - lo) * scale))
        chars.append(ASCII_LEVELS[max(0, min(scale, idx))])
    return "".join(chars)


def format_float(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def render_dashboard(
    run_dir: Path,
    metrics_path: Path,
    results_path: Path,
    crash_path: Path,
    events: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    points: int,
) -> str:
    width = shutil.get_terminal_size((100, 30)).columns
    chart_width = max(20, width - 36)
    start_event = last_event(events, "run_start") or {}
    train_event = last_event(events, "train") or {}
    val_event = last_event(events, "val") or {}
    summary_event = last_event(events, "summary") or {}
    results = load_json(results_path) if results_path.is_file() else None
    crash = load_json(crash_path) if crash_path.is_file() else None

    lines = [
        f"Run Dir: {run_dir}",
        f"Metrics: {metrics_path}",
        f"Run ID: {start_event.get('run_id', summary_event.get('run_id', '-'))}",
        f"Mode: {start_event.get('mode', summary_event.get('mode', '-'))}",
        f"Status: {results.get('status') if results else ('failed' if crash else 'running')}",
    ]
    if train_event:
        lines.append(
            "Last Train: step={} loss={} matrix_lr={} step_seconds={}".format(
                train_event.get("step", "-"),
                format_float(train_event.get("train_loss")),
                format_float(train_event.get("matrix_lr")),
                format_float(train_event.get("step_seconds")),
            )
        )
    if val_event:
        lines.append(
            "Last Val: step={} phase={} val_loss={} val_bpb={}".format(
                val_event.get("step", "-"),
                val_event.get("phase", "-"),
                format_float(val_event.get("val_loss")),
                format_float(val_event.get("val_bpb")),
            )
        )
    if results:
        lines.append(
            "Result: val_bpb={} artifact_bytes={} total_seconds={}".format(
                format_float(results.get("val_bpb")),
                results.get("artifact_bytes", "-"),
                format_float(results.get("total_seconds")),
            )
        )
    if crash:
        lines.append(f"Crash: {crash.get('error_type', 'error')} {crash.get('error_message', '')}")
    lines.append("")

    for metric_name in metric_names:
        series = metric_series(events, metric_name, points)
        last_value = series[-1] if series else None
        chart = sparkline(series, chart_width)
        lines.append(f"{metric_name:<12} last={format_float(last_value):>12}  {chart}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously render a run's metrics.jsonl stream in the terminal.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Run dir, metrics.jsonl, results.json, or latest.json pointer.")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--points", type=int, default=80)
    parser.add_argument("--refresh_seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no_clear", action="store_true")
    parser.add_argument("--exit_when_complete", action="store_true")
    args = parser.parse_args()

    metric_names = parse_metric_names(args.metrics)
    run_dir, metrics_path, results_path, crash_path = resolve_run_paths(args.path)

    while True:
        events = load_jsonl(metrics_path)
        output = render_dashboard(run_dir, metrics_path, results_path, crash_path, events, metric_names, args.points)
        if not args.no_clear and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        done = results_path.is_file() or crash_path.is_file()
        if args.once or (args.exit_when_complete and done):
            break
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    main()
