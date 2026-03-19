from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PATH = "./runs/autoresearch_5090/index/latest.json"
DEFAULT_METRICS = ("train_loss", "val_bpb", "matrix_lr", "tokens_per_second")
ASCII_LEVELS = " .:-=+*#%@"
METRIC_FIELDS = {
    "train_loss": ("train", "train_loss"),
    "matrix_lr": ("train", "matrix_lr"),
    "embed_lr": ("train", "embed_lr"),
    "head_lr": ("train", "head_lr"),
    "scalar_lr": ("train", "scalar_lr"),
    "step_seconds": ("train", "step_seconds"),
    "tokens_per_second": ("train", None),
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
    if not path.exists() and path.name == "latest.json":
        active_path = path.with_name("active.json")
        if active_path.is_file():
            path = active_path
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
        crash_path = Path(payload.get("crash_path", run_dir / "crash.json"))
        return run_dir, metrics_path, results_path, crash_path
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


def metric_value(event: dict[str, Any], metric_name: str) -> float | None:
    _event_name, field_name = METRIC_FIELDS[metric_name]
    if metric_name == "tokens_per_second":
        total_tokens = event.get("total_tokens")
        elapsed = event.get("elapsed_training_seconds")
        if isinstance(total_tokens, (int, float)) and isinstance(elapsed, (int, float)) and float(elapsed) > 0.0:
            return float(total_tokens) / float(elapsed)
        return None
    value = event.get(field_name) if field_name is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_series(events: list[dict[str, Any]], metric_name: str, limit: int) -> list[float]:
    event_name, _field_name = METRIC_FIELDS[metric_name]
    values: list[float] = []
    for event in events:
        if event.get("event") != event_name:
            continue
        value = metric_value(event, metric_name)
        if value is not None:
            values.append(value)
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


def sample_series(values: list[float], width: int) -> list[float]:
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return values
    step = len(values) / width
    return [values[min(int(i * step), len(values) - 1)] for i in range(width)]


def render_line_plot(values: list[float], width: int, height: int) -> list[str]:
    plot_width = max(8, width)
    plot_height = max(3, height)
    if not values:
        return [" " * plot_width for _ in range(plot_height)]

    sampled = sample_series(values, plot_width)
    lo = min(sampled)
    hi = max(sampled)
    if hi <= lo:
        row_positions = [plot_height // 2 for _ in sampled]
    else:
        row_positions = [
            int(round((1.0 - ((value - lo) / (hi - lo))) * (plot_height - 1)))
            for value in sampled
        ]

    grid = [[" " for _ in range(len(sampled))] for _ in range(plot_height)]
    prev_row: int | None = None
    prev_col: int | None = None
    for col, row in enumerate(row_positions):
        grid[row][col] = "*"
        if prev_row is not None and prev_col is not None:
            gap = col - prev_col
            if gap > 1:
                for delta in range(1, gap):
                    interp_col = prev_col + delta
                    interp_row = int(round(prev_row + (row - prev_row) * (delta / gap)))
                    if grid[interp_row][interp_col] == " ":
                        grid[interp_row][interp_col] = "."
        prev_row = row
        prev_col = col
    return ["".join(row) for row in grid]


def format_float(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def compact_path(path: Path, max_len: int) -> str:
    text = str(path)
    if len(text) <= max_len:
        return text
    keep = max(8, max_len - 3)
    return "..." + text[-keep:]


def current_status(results: dict[str, Any] | None, crash: dict[str, Any] | None) -> str:
    if results is not None:
        return str(results.get("status", "success"))
    if crash is not None:
        return "failed"
    return "running"


def find_results_tsv_path(run_dir: Path) -> Path | None:
    for parent in [run_dir] + list(run_dir.parents[:5]):
        candidate = parent / "results.tsv"
        if candidate.is_file():
            return candidate
    return None


def load_results_rows(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows[-limit:]


def parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def recent_run_series(rows: list[dict[str, str]], key: str, limit: int) -> list[float]:
    values: list[float] = []
    for row in rows[-limit:]:
        value = parse_optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def recent_run_rows(rows: list[dict[str, str]], limit: int, width: int) -> list[str]:
    if not rows:
        return ["(no completed runs yet)"]
    shown = rows[-limit:]
    bpb_values = [parse_optional_float(row.get("val_bpb")) for row in rows]
    best_bpb = min(value for value in bpb_values if value is not None) if any(value is not None for value in bpb_values) else None
    rendered: list[str] = []
    for row in reversed(shown):
        val_bpb = parse_optional_float(row.get("val_bpb"))
        artifact_bytes = row.get("artifact_bytes", "null")
        training_seconds = row.get("training_seconds", "null")
        best_marker = " *best*" if best_bpb is not None and val_bpb is not None and abs(val_bpb - best_bpb) < 1e-12 else ""
        rendered.append(
            "run_id={} val_bpb={} artifact_bytes={} train_s={}{}".format(
                str(row.get("run_id", ""))[:24],
                "null" if val_bpb is None else f"{val_bpb:.6f}",
                artifact_bytes,
                training_seconds,
                best_marker,
            )[:width]
        )
    return rendered


def collect_run_state(
    run_dir: Path,
    metrics_path: Path,
    results_path: Path,
    crash_path: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    start_event = last_event(events, "run_start") or {}
    train_event = last_event(events, "train") or {}
    val_event = last_event(events, "val") or {}
    summary_event = last_event(events, "summary") or {}
    results = load_json(results_path) if results_path.is_file() else None
    crash = load_json(crash_path) if crash_path.is_file() else None
    results_tsv_path = find_results_tsv_path(run_dir)
    return {
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "results_path": results_path,
        "crash_path": crash_path,
        "results_tsv_path": results_tsv_path,
        "events": events,
        "start_event": start_event,
        "train_event": train_event,
        "val_event": val_event,
        "summary_event": summary_event,
        "results": results,
        "crash": crash,
        "status": current_status(results, crash),
    }


def recent_event_rows(events: list[dict[str, Any]], limit: int) -> list[str]:
    rows: list[str] = []
    for event in reversed(events):
        kind = event.get("event")
        if kind == "train":
            rows.append(
                "step={} train loss={} lr={} tok/s={}".format(
                    event.get("step", "-"),
                    format_float(event.get("train_loss")),
                    format_float(event.get("matrix_lr")),
                    format_float(metric_value(event, "tokens_per_second")),
                )
            )
        elif kind == "val":
            rows.append(
                "step={} val/{} loss={} bpb={}".format(
                    event.get("step", "-"),
                    event.get("phase", "-"),
                    format_float(event.get("val_loss")),
                    format_float(event.get("val_bpb")),
                )
            )
        elif kind == "summary":
            rows.append(
                "summary status={} val_bpb={} total_seconds={}".format(
                    event.get("status", "-"),
                    format_float(event.get("val_bpb")),
                    format_float(event.get("total_seconds")),
                )
            )
        elif kind == "crash":
            rows.append(
                "crash {} {}".format(
                    event.get("error_type", "error"),
                    str(event.get("error_message", ""))[:80],
                )
            )
        if len(rows) >= limit:
            break
    if not rows:
        rows.append("(no events yet)")
    return rows


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
    state = collect_run_state(run_dir, metrics_path, results_path, crash_path, events)
    start_event = state["start_event"]
    train_event = state["train_event"]
    val_event = state["val_event"]
    results = state["results"]
    crash = state["crash"]

    lines = [
        f"Run Dir: {run_dir}",
        f"Metrics: {metrics_path}",
        f"Run ID: {start_event.get('run_id', '-')}",
        f"Mode: {start_event.get('mode', state['summary_event'].get('mode', '-'))}",
        f"Status: {state['status']}",
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
        lines.append(f"{metric_name:<16} last={format_float(last_value):>12}")
        for plot_line in render_line_plot(series, chart_width, 4):
            lines.append(plot_line)
        lines.append("")

    recent_runs = []
    if state["results_tsv_path"] is not None:
        recent_runs = load_results_rows(state["results_tsv_path"], limit=12)
    if recent_runs:
        lines.append("Recent Run History (val_bpb, lower is better)")
        for plot_line in render_line_plot(recent_run_series(recent_runs, "val_bpb", 12), chart_width, 4):
            lines.append(plot_line)
        lines.extend(recent_run_rows(recent_runs, limit=5, width=width))
    return "\n".join(lines)


def render_tui_lines(state: dict[str, Any], metric_names: tuple[str, ...], points: int, width: int, height: int) -> list[str]:
    start_event = state["start_event"]
    train_event = state["train_event"]
    val_event = state["val_event"]
    results = state["results"]
    crash = state["crash"]
    events = state["events"]

    usable_width = max(60, width - 1)
    num_cols = 2 if usable_width >= 96 and len(metric_names) > 1 else 1
    gap = 3
    panel_width = (usable_width - (gap * (num_cols - 1))) // num_cols
    chart_height = 6 if height >= 34 else 4
    lines = [
        "autoresearch-parameter-golf monitor  (q to quit)",
        f"Run: {start_event.get('run_id', '-')}",
        f"Dir: {compact_path(state['run_dir'], usable_width - 5)}",
        "Status: {}   Mode: {}   Step: {}".format(
            state["status"],
            start_event.get("mode", state["summary_event"].get("mode", "-")),
            train_event.get("step", val_event.get("step", state["summary_event"].get("step", "-"))),
        ),
    ]

    top_metrics = [
        "Train Loss: {}".format(format_float(train_event.get("train_loss"))),
        "Val BPB: {}".format(format_float((results or {}).get("val_bpb", val_event.get("val_bpb")))),
        "Matrix LR: {}".format(format_float(train_event.get("matrix_lr"))),
        "Tok/s: {}".format(format_float(metric_value(train_event, "tokens_per_second"))),
        "Step s: {}".format(format_float(train_event.get("step_seconds"))),
        "Artifact: {}".format((results or {}).get("artifact_bytes", "-")),
    ]
    lines.append(" | ".join(top_metrics))
    if val_event:
        lines.append(
            "Last Val: step={} phase={} loss={} bpb={}".format(
                val_event.get("step", "-"),
                val_event.get("phase", "-"),
                format_float(val_event.get("val_loss")),
                format_float(val_event.get("val_bpb")),
            )
        )
    if results:
        lines.append(
            "Result: total_seconds={} peak_vram_mb={}".format(
                format_float(results.get("total_seconds")),
                format_float(results.get("peak_vram_mb")),
            )
        )
    if crash:
        lines.append("Crash: {} {}".format(crash.get("error_type", "error"), str(crash.get("error_message", ""))[: usable_width - 8]))
    lines.append("-" * usable_width)

    metric_panels: list[list[str]] = []
    for metric_name in metric_names:
        series = metric_series(events, metric_name, points)
        last_value = series[-1] if series else None
        lo = min(series) if series else None
        hi = max(series) if series else None
        panel = [
            "{} last={} min={} max={}".format(
                metric_name,
                format_float(last_value),
                format_float(lo),
                format_float(hi),
            )[:panel_width].ljust(panel_width),
        ]
        plot_lines = render_line_plot(series, max(8, panel_width), chart_height)
        panel.extend(line[:panel_width].ljust(panel_width) for line in plot_lines)
        metric_panels.append(panel)

    panel_height = 1 + chart_height
    for row_start in range(0, len(metric_panels), num_cols):
        row_panels = metric_panels[row_start : row_start + num_cols]
        while len(row_panels) < num_cols:
            row_panels.append([" " * panel_width for _ in range(panel_height)])
        for line_idx in range(panel_height):
            lines.append((" " * gap).join(panel[line_idx] for panel in row_panels)[:usable_width])
        lines.append("")

    recent_runs = []
    if state["results_tsv_path"] is not None:
        recent_runs = load_results_rows(state["results_tsv_path"], limit=16)

    remaining_lines = max(0, height - len(lines))
    if recent_runs and remaining_lines >= 8:
        reserve_for_events = 4
        history_space = max(0, remaining_lines - reserve_for_events)
        history_height = min(6 if height >= 34 else 4, max(2, history_space - 4))
        history_row_limit = max(1, min(3, history_space - 2 - history_height))
        lines.append("-" * usable_width)
        lines.append("Recent Run History (val_bpb, lower is better)")
        for plot_line in render_line_plot(
            recent_run_series(recent_runs, "val_bpb", max(8, min(16, points))),
            usable_width,
            history_height,
        ):
            lines.append(plot_line[:usable_width].ljust(usable_width))
        lines.extend(recent_run_rows(recent_runs, limit=history_row_limit, width=usable_width))
        lines.append("")

    remaining_lines = max(0, height - len(lines))
    event_row_limit = max(1, min(6, remaining_lines - 2))
    lines.append("-" * usable_width)
    lines.append("Recent Events")
    for row in recent_event_rows(events, event_row_limit):
        lines.append(row)

    return [line[:usable_width] for line in lines[: max(1, height)]]


def run_plain_loop(
    run_dir: Path,
    metrics_path: Path,
    results_path: Path,
    crash_path: Path,
    metric_names: tuple[str, ...],
    points: int,
    refresh_seconds: float,
    once: bool,
    no_clear: bool,
    exit_when_complete: bool,
) -> None:
    while True:
        events = load_jsonl(metrics_path)
        output = render_dashboard(run_dir, metrics_path, results_path, crash_path, events, metric_names, points)
        if not no_clear and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        done = results_path.is_file() or crash_path.is_file()
        if once or (exit_when_complete and done):
            break
        time.sleep(refresh_seconds)


def run_tui_loop(
    run_dir: Path,
    metrics_path: Path,
    results_path: Path,
    crash_path: Path,
    metric_names: tuple[str, ...],
    points: int,
    refresh_seconds: float,
    exit_when_complete: bool,
) -> None:
    import curses

    def _main(stdscr: Any) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(max(50, int(refresh_seconds * 1000)))
        while True:
            events = load_jsonl(metrics_path)
            state = collect_run_state(run_dir, metrics_path, results_path, crash_path, events)
            height, width = stdscr.getmaxyx()
            lines = render_tui_lines(state, metric_names, points, width, height)
            stdscr.erase()
            for row_idx, line in enumerate(lines[:height]):
                try:
                    stdscr.addnstr(row_idx, 0, line, max(0, width - 1))
                except curses.error:
                    pass
            stdscr.refresh()
            done = results_path.is_file() or crash_path.is_file()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            if exit_when_complete and done:
                break

    curses.wrapper(_main)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor a run's metrics.jsonl stream in a remote-friendly terminal UI.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Run dir, metrics.jsonl, results.json, or latest.json pointer.")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--points", type=int, default=80)
    parser.add_argument("--refresh_seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--no_clear", action="store_true")
    parser.add_argument("--exit_when_complete", action="store_true")
    args = parser.parse_args()

    metric_names = parse_metric_names(args.metrics)
    run_dir, metrics_path, results_path, crash_path = resolve_run_paths(args.path)
    use_plain = args.plain or args.once or not sys.stdout.isatty()

    if use_plain:
        run_plain_loop(
            run_dir=run_dir,
            metrics_path=metrics_path,
            results_path=results_path,
            crash_path=crash_path,
            metric_names=metric_names,
            points=args.points,
            refresh_seconds=args.refresh_seconds,
            once=args.once,
            no_clear=args.no_clear,
            exit_when_complete=args.exit_when_complete,
        )
        return

    try:
        run_tui_loop(
            run_dir=run_dir,
            metrics_path=metrics_path,
            results_path=results_path,
            crash_path=crash_path,
            metric_names=metric_names,
            points=args.points,
            refresh_seconds=args.refresh_seconds,
            exit_when_complete=args.exit_when_complete,
        )
    except Exception:
        run_plain_loop(
            run_dir=run_dir,
            metrics_path=metrics_path,
            results_path=results_path,
            crash_path=crash_path,
            metric_names=metric_names,
            points=args.points,
            refresh_seconds=args.refresh_seconds,
            once=args.once,
            no_clear=args.no_clear,
            exit_when_complete=args.exit_when_complete,
        )


if __name__ == "__main__":
    main()
