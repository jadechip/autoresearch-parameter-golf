from __future__ import annotations

import argparse
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_state_dir() -> Path:
    return repo_root() / ".autoresearch"


def runs_dir(state_dir: Path) -> Path:
    return state_dir / "runs"


def activity_log_path(state_dir: Path) -> Path:
    return state_dir / "activity.log"


def newest_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def tail_lines(path: Path | None, limit: int) -> list[str]:
    if path is None or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def render_snapshot(state_dir: Path, history_lines: int) -> list[str]:
    run_logs_dir = runs_dir(state_dir)
    latest_log = newest_file(run_logs_dir, "codex-iteration-*.log")
    latest_last = newest_file(run_logs_dir, "codex-iteration-*.last.txt")

    lines = [
        "autoresearch-parameter-golf codex-loop watcher",
        f"State Dir: {state_dir}",
    ]

    activity_lines = tail_lines(activity_log_path(state_dir), history_lines)
    lines.append("Activity")
    lines.extend(activity_lines if activity_lines else ["(no activity yet)"])

    lines.append("")
    lines.append(f"Current Iteration Log: {latest_log if latest_log is not None else '(none yet)'}")
    lines.extend(tail_lines(latest_log, history_lines) if latest_log is not None else ["(no iteration log yet)"])

    lines.append("")
    lines.append(f"Latest Summary: {latest_last if latest_last is not None else '(none yet)'}")
    lines.extend(tail_lines(latest_last, history_lines) if latest_last is not None else ["(no last-message summary yet)"])
    return lines


def read_new_lines(path: Path | None, offset: int) -> tuple[int, list[str]]:
    if path is None or not path.is_file():
        return 0, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        new_offset = handle.tell()
    if not chunk:
        return new_offset, []
    return new_offset, chunk.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the Ralph-style Codex autoresearch loop.")
    parser.add_argument("--state_dir", type=str, default=str(default_state_dir()))
    parser.add_argument("--poll_seconds", type=float, default=1.0)
    parser.add_argument("--history_lines", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    snapshot = render_snapshot(state_dir, args.history_lines)
    print("\n".join(snapshot), flush=True)
    if args.once:
        return

    activity_path = activity_log_path(state_dir)
    latest_log = newest_file(runs_dir(state_dir), "codex-iteration-*.log")
    latest_last = newest_file(runs_dir(state_dir), "codex-iteration-*.last.txt")
    activity_offset = activity_path.stat().st_size if activity_path.is_file() else 0
    log_offset = latest_log.stat().st_size if latest_log is not None and latest_log.exists() else 0
    last_offset = latest_last.stat().st_size if latest_last is not None and latest_last.exists() else 0

    while True:
        new_log = newest_file(runs_dir(state_dir), "codex-iteration-*.log")
        if new_log != latest_log:
            latest_log = new_log
            log_offset = 0
            if latest_log is not None:
                print(f"\n[codex] switched_to {latest_log}", flush=True)

        new_last = newest_file(runs_dir(state_dir), "codex-iteration-*.last.txt")
        if new_last != latest_last:
            latest_last = new_last
            last_offset = 0
            if latest_last is not None:
                print(f"\n[summary] switched_to {latest_last}", flush=True)

        activity_offset, activity_lines = read_new_lines(activity_path, activity_offset)
        for line in activity_lines:
            print(f"[activity] {line}", flush=True)

        log_offset, log_lines = read_new_lines(latest_log, log_offset)
        for line in log_lines:
            print(f"[codex] {line}", flush=True)

        last_offset, summary_lines = read_new_lines(latest_last, last_offset)
        for line in summary_lines:
            print(f"[summary] {line}", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
