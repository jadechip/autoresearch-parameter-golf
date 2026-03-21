from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from train import append_jsonl, atomic_write_json, load_and_validate_results


SESSION_SCHEMA_VERSION = "pgolf.autoresearch_session.v1"
EXPERIMENT_LOG_SCHEMA_VERSION = "pgolf.autoresearch_experiment.v1"
SEARCH_MIN_MEANINGFUL_BPB_GAIN = 0.001
SEARCH_ARTIFACT_TARGET_BYTES_MIN = 12_000_000
SEARCH_ARTIFACT_TARGET_BYTES_MAX = 15_500_000
SEARCH_MAX_CONSECUTIVE_MICRO_EXPERIMENTS = 3
SEARCH_STRUCTURAL_AXES = [
    "d_model",
    "shared_layers_vs_recurrence_loops",
    "tail_layers",
    "mlp_mult",
    "adapter_rank_and_targets",
    "fake_quant_start_step_and_clip_percentile",
]
SEARCH_EXTERNAL_PRIORS = [
    "The recovered compact 5090 baseline is now in the wrong regime for a likely win: the next serious search should move toward a stronger, near-full-budget carrier instead of more shrinkage.",
    "Top leaderboard entries cluster around near-full-budget nonrecurrent carriers, mixed low-bit quantization, longer context, and stronger training/eval tricks rather than pure recurrence.",
    "Low-rank Q is a promising structural reallocation tool because it can buy more unique depth, a stronger local module, or more steps.",
    "Late selective coarse-group quantization is more promising than blanket early QAT or highly heterogeneous mixed-bit schemes.",
    "Longer context and sliding-window eval appear promising, but only if they remain competition-valid and fit the fixed budget.",
    "Optimizer bundles with Muon momentum, weight decay, warmdown, and possibly SWA should follow a stronger structural candidate, not replace one.",
]
SEARCH_NEXT_PRIORITY_AXES = [
    "near-full-budget carrier with lower recurrence, more unique depth, and wider MLPs",
    "low-rank Q as a structural reallocation tool",
    "late selective coarse-group quantization or post-quant checkpoint soup",
    "selective higher precision for embeddings or head",
    "compute-aware batch and context curricula",
    "smarter local-token modules with better byte/quality tradeoffs",
]
SEARCH_DO_NOT_OVERWEIGHT = [
    "Do not cargo-cult leaderboard entries.",
    "Treat public leaderboard patterns as priors, not recipes.",
    "Do not repeat known-losing compact-line moves without a new major hypothesis.",
    "Do not keep using recurrence or tiny-model compression as the main search direction.",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_state_dir() -> Path:
    return repo_root() / ".autoresearch"


def session_path(state_dir: Path) -> Path:
    return state_dir / "session.json"


def experiments_path(state_dir: Path) -> Path:
    return state_dir / "experiments.jsonl"


def notes_path(state_dir: Path) -> Path:
    return state_dir / "notes.md"


def activity_log_path(state_dir: Path) -> Path:
    return state_dir / "activity.log"


def errors_log_path(state_dir: Path) -> Path:
    return state_dir / "errors.log"


def runs_dir_path(state_dir: Path) -> Path:
    return state_dir / "runs"


def git_output(args: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        args,
        cwd=repo_root() if cwd is None else cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def current_branch() -> str:
    return git_output(["git", "branch", "--show-current"])


def current_commit() -> str:
    return git_output(["git", "rev-parse", "HEAD"])


def current_commit_short() -> str:
    return git_output(["git", "rev-parse", "--short", "HEAD"])


def resolve_results_source(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indexed_source = payload.get("indexed_source_results_path")
    if indexed_source:
        return Path(str(indexed_source))
    return path


def load_session(state_dir: Path) -> dict[str, Any]:
    path = session_path(state_dir)
    if not path.is_file():
        raise ValueError(f"missing autoresearch session file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise ValueError(f"unexpected autoresearch session schema: {payload.get('schema_version')}")
    return payload


def write_session(state_dir: Path, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload["schema_version"] = SESSION_SCHEMA_VERSION
    payload["updated_at_unix"] = time.time()
    return atomic_write_json(session_path(state_dir), payload)


def append_experiment_event(state_dir: Path, payload: dict[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": EXPERIMENT_LOG_SCHEMA_VERSION,
        "event_time_unix": time.time(),
        **payload,
    }
    return append_jsonl(experiments_path(state_dir), event)


def ensure_notes_file(state_dir: Path) -> None:
    path = notes_path(state_dir)
    if path.exists():
        return
    path.write_text(
        "# Autoresearch Notes\n\n"
        "Accepted state:\n"
        "- The accepted code state is the current branch tip plus `session.json`.\n"
        "- Numeric `best.json` is telemetry only; it may point at a reverted run.\n\n"
        "Current campaign:\n"
        f"- Soft artifact target band for 5090 search: {SEARCH_ARTIFACT_TARGET_BYTES_MIN:,} to {SEARCH_ARTIFACT_TARGET_BYTES_MAX:,} bytes.\n"
        f"- Meaningful improvement threshold: about {SEARCH_MIN_MEANINGFUL_BPB_GAIN:.3f} val_bpb.\n"
        f"- Do not spend more than {SEARCH_MAX_CONSECUTIVE_MICRO_EXPERIMENTS} consecutive losing micro-tuning runs without a structural / byte-allocation experiment.\n\n"
        "Structural campaign checklist:\n"
        "- [ ] d_model\n"
        "- [ ] shared_layers vs recurrence_loops\n"
        "- [ ] tail_layers\n"
        "- [ ] mlp_mult\n"
        "- [ ] adapter_rank / adapter_targets\n"
        "- [ ] fake_quant_start_step / clip_percentile\n\n"
        "Frontier priors:\n"
        "- Spend bytes deliberately if the accepted line is still well below the hard `16 MB` cap.\n"
        "- Prefer a stronger near-full-budget carrier over more recurrence or more shrinkage.\n"
        "- Low-rank Q, selective quantization, and compute-aware context are higher-priority than local tail-width retunes.\n"
        "- Longer context or sliding-window eval should only be tried when compute is reclaimed elsewhere.\n"
        "- Avoid repeating known-losing compact-line moves without a new major hypothesis.\n\n"
        "Hypothesis ledger:\n"
        "- Open:\n"
        "- Tried:\n"
        "- Won:\n"
        "- Rejected:\n",
        encoding="utf-8",
    )


def ensure_loop_support_files(state_dir: Path) -> None:
    runs_dir_path(state_dir).mkdir(parents=True, exist_ok=True)
    for path in (activity_log_path(state_dir), errors_log_path(state_dir)):
        if not path.exists():
            path.write_text("", encoding="utf-8")


def init_session(state_dir: Path, baseline_results_path: Path, force: bool = False) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = session_path(state_dir)
    if path.exists() and not force:
        raise ValueError(f"autoresearch session already exists: {path}")

    resolved_results = resolve_results_source(baseline_results_path)
    baseline = dict(load_and_validate_results(resolved_results))
    if baseline.get("status") != "success":
        raise ValueError("baseline results must have status=success")

    branch = current_branch()
    commit = current_commit()
    commit_short = current_commit_short()
    now = time.time()
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "ready",
        "created_at_unix": now,
        "updated_at_unix": now,
        "repo_root": str(repo_root()),
        "state_dir": str(state_dir),
        "current_branch": branch,
        "accepted_commit": commit,
        "accepted_commit_short": commit_short,
        "accepted_run_id": baseline["run_id"],
        "accepted_val_bpb": baseline["val_bpb"],
        "accepted_artifact_bytes": baseline["artifact_bytes"],
        "accepted_results_path": str(resolved_results),
        "baseline_run_id": baseline["run_id"],
        "baseline_val_bpb": baseline["val_bpb"],
        "baseline_artifact_bytes": baseline["artifact_bytes"],
        "baseline_results_path": str(resolved_results),
        "latest_run_id": baseline["run_id"],
        "latest_results_path": str(resolved_results),
        "latest_status": baseline["status"],
        "latest_val_bpb": baseline["val_bpb"],
        "latest_artifact_bytes": baseline["artifact_bytes"],
        "latest_decision": "accepted",
        "current_experiment": None,
        "search_policy": {
            "min_meaningful_bpb_gain": SEARCH_MIN_MEANINGFUL_BPB_GAIN,
            "artifact_target_bytes_min": SEARCH_ARTIFACT_TARGET_BYTES_MIN,
            "artifact_target_bytes_max": SEARCH_ARTIFACT_TARGET_BYTES_MAX,
            "max_consecutive_losing_micro_experiments": SEARCH_MAX_CONSECUTIVE_MICRO_EXPERIMENTS,
            "structural_axes": list(SEARCH_STRUCTURAL_AXES),
            "external_priors": list(SEARCH_EXTERNAL_PRIORS),
            "next_priority_axes": list(SEARCH_NEXT_PRIORITY_AXES),
            "do_not_overweight": list(SEARCH_DO_NOT_OVERWEIGHT),
        },
    }
    write_session(state_dir, session)
    ensure_notes_file(state_dir)
    ensure_loop_support_files(state_dir)
    append_experiment_event(
        state_dir,
        {
            "event": "baseline_init",
            "run_id": baseline["run_id"],
            "results_path": str(resolved_results),
            "branch": branch,
            "commit": commit,
            "commit_short": commit_short,
            "status": baseline["status"],
            "val_bpb": baseline["val_bpb"],
            "artifact_bytes": baseline["artifact_bytes"],
            "training_seconds": baseline["training_seconds"],
            "decision": "accepted",
        },
    )
    return load_session(state_dir)


def require_ready(state_dir: Path) -> dict[str, Any]:
    session = load_session(state_dir)
    if session.get("status") != "ready":
        raise ValueError(
            f"autoresearch session is not ready: status={session.get('status')}. "
            f"Current experiment: {session.get('current_experiment')}"
        )
    return session


def start_run(
    state_dir: Path,
    *,
    run_id: str,
    output_dir: str,
    results_path_value: str,
    metrics_path: str,
    tensorboard_log_dir: str | None,
    crash_path: str,
) -> dict[str, Any]:
    session = require_ready(state_dir)
    experiment = {
        "run_id": run_id,
        "output_dir": output_dir,
        "results_path": results_path_value,
        "metrics_path": metrics_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "crash_path": crash_path,
        "started_at_unix": time.time(),
        "branch": current_branch(),
        "commit": current_commit(),
        "commit_short": current_commit_short(),
    }
    session["status"] = "running"
    session["current_branch"] = experiment["branch"]
    session["current_experiment"] = experiment
    write_session(state_dir, session)
    append_experiment_event(
        state_dir,
        {
            "event": "run_start",
            **experiment,
        },
    )
    return load_session(state_dir)


def finish_run(state_dir: Path, results_json: Path) -> dict[str, Any]:
    session = load_session(state_dir)
    resolved_results = resolve_results_source(results_json)
    results = dict(load_and_validate_results(resolved_results))
    branch = current_branch()
    commit = current_commit()
    commit_short = current_commit_short()
    append_experiment_event(
        state_dir,
        {
            "event": "run_result",
            "run_id": results["run_id"],
            "results_path": str(resolved_results),
            "branch": branch,
            "commit": commit,
            "commit_short": commit_short,
            "status": results["status"],
            "val_bpb": results["val_bpb"],
            "artifact_bytes": results["artifact_bytes"],
            "training_seconds": results["training_seconds"],
            "total_seconds": results["total_seconds"],
            "output_dir": results["output_dir"],
        },
    )
    session["status"] = "ready"
    session["current_branch"] = branch
    session["latest_run_id"] = results["run_id"]
    session["latest_results_path"] = str(resolved_results)
    session["latest_status"] = results["status"]
    session["latest_val_bpb"] = results["val_bpb"]
    session["latest_artifact_bytes"] = results["artifact_bytes"]
    session["current_experiment"] = None
    write_session(state_dir, session)
    return load_session(state_dir)


def abort_run(state_dir: Path, run_id: str, reason: str) -> dict[str, Any]:
    session = load_session(state_dir)
    append_experiment_event(
        state_dir,
        {
            "event": "run_abort",
            "run_id": run_id,
            "branch": current_branch(),
            "commit": current_commit(),
            "commit_short": current_commit_short(),
            "reason": reason,
        },
    )
    session["status"] = "ready"
    session["current_experiment"] = None
    write_session(state_dir, session)
    return load_session(state_dir)


def decide_run(state_dir: Path, run_id: str, decision: str, results_json: Path | None = None) -> dict[str, Any]:
    session = load_session(state_dir)
    if decision not in {"accepted", "reverted"}:
        raise ValueError(f"unsupported decision: {decision}")

    resolved_results = None if results_json is None else resolve_results_source(results_json)
    results = None if resolved_results is None else dict(load_and_validate_results(resolved_results))
    branch = current_branch()
    commit = current_commit()
    commit_short = current_commit_short()

    append_experiment_event(
        state_dir,
        {
            "event": "decision",
            "run_id": run_id,
            "decision": decision,
            "branch": branch,
            "commit": commit,
            "commit_short": commit_short,
            "results_path": None if resolved_results is None else str(resolved_results),
            "val_bpb": None if results is None else results.get("val_bpb"),
        },
    )

    session["current_branch"] = branch
    session["latest_decision"] = decision
    if decision == "accepted":
        session["accepted_commit"] = commit
        session["accepted_commit_short"] = commit_short
        session["accepted_run_id"] = run_id
        if results is not None:
            session["accepted_val_bpb"] = results["val_bpb"]
            session["accepted_artifact_bytes"] = results["artifact_bytes"]
            session["accepted_results_path"] = str(resolved_results)
    write_session(state_dir, session)
    return load_session(state_dir)


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Ralph-style autoresearch session state.")
    parser.add_argument("--state_dir", type=str, default=str(default_state_dir()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--baseline_results", type=str, required=True)
    init_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("require-ready")
    subparsers.add_parser("show")

    start_parser = subparsers.add_parser("start-run")
    start_parser.add_argument("--run_id", type=str, required=True)
    start_parser.add_argument("--output_dir", type=str, required=True)
    start_parser.add_argument("--results_path", type=str, required=True)
    start_parser.add_argument("--metrics_path", type=str, required=True)
    start_parser.add_argument("--tensorboard_log_dir", type=str, default=None)
    start_parser.add_argument("--crash_path", type=str, required=True)

    finish_parser = subparsers.add_parser("finish-run")
    finish_parser.add_argument("--results_json", type=str, required=True)

    abort_parser = subparsers.add_parser("abort-run")
    abort_parser.add_argument("--run_id", type=str, required=True)
    abort_parser.add_argument("--reason", type=str, default="aborted")

    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--run_id", type=str, required=True)
    decide_parser.add_argument("--decision", type=str, choices=("accepted", "reverted"), required=True)
    decide_parser.add_argument("--results_json", type=str, default=None)

    args = parser.parse_args()
    state_dir = Path(args.state_dir)

    if args.command == "init":
        print_payload(init_session(state_dir, Path(args.baseline_results), force=args.force))
        return
    if args.command == "require-ready":
        print_payload(require_ready(state_dir))
        return
    if args.command == "show":
        print_payload(load_session(state_dir))
        return
    if args.command == "start-run":
        print_payload(
            start_run(
                state_dir,
                run_id=args.run_id,
                output_dir=args.output_dir,
                results_path_value=args.results_path,
                metrics_path=args.metrics_path,
                tensorboard_log_dir=args.tensorboard_log_dir,
                crash_path=args.crash_path,
            )
        )
        return
    if args.command == "finish-run":
        print_payload(finish_run(state_dir, Path(args.results_json)))
        return
    if args.command == "abort-run":
        print_payload(abort_run(state_dir, args.run_id, args.reason))
        return
    if args.command == "decide":
        print_payload(decide_run(state_dir, args.run_id, args.decision, None if args.results_json is None else Path(args.results_json)))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
