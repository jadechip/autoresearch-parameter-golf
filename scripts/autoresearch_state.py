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
TRACKED_ACCEPTED_STATE_SCHEMA_VERSION = "pgolf.autoresearch_git_state.v1"
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


def tracked_autoresearch_dir() -> Path:
    return repo_root() / "state" / "autoresearch"


def promoted_configs_dir() -> Path:
    return repo_root() / "configs" / "promoted"


def tracked_accepted_state_path() -> Path:
    return tracked_autoresearch_dir() / "accepted_state.json"


def tracked_promoted_5090_config_path() -> Path:
    return promoted_configs_dir() / "autoresearch_5090_best.json"


def tracked_promoted_h100_config_path(mode: str) -> Path:
    if mode not in {"1x", "8x"}:
        raise ValueError(f"unsupported promoted h100 mode: {mode}")
    return promoted_configs_dir() / f"autoresearch_h100_{mode}_best.json"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_results_source(path: Path) -> Path:
    payload = load_json(path)
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


def repo_relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def default_search_policy() -> dict[str, Any]:
    return {
        "min_meaningful_bpb_gain": SEARCH_MIN_MEANINGFUL_BPB_GAIN,
        "artifact_target_bytes_min": SEARCH_ARTIFACT_TARGET_BYTES_MIN,
        "artifact_target_bytes_max": SEARCH_ARTIFACT_TARGET_BYTES_MAX,
        "max_consecutive_losing_micro_experiments": SEARCH_MAX_CONSECUTIVE_MICRO_EXPERIMENTS,
        "structural_axes": list(SEARCH_STRUCTURAL_AXES),
        "external_priors": list(SEARCH_EXTERNAL_PRIORS),
        "next_priority_axes": list(SEARCH_NEXT_PRIORITY_AXES),
        "do_not_overweight": list(SEARCH_DO_NOT_OVERWEIGHT),
    }


def portable_5090_defaults() -> dict[str, Any]:
    return {
        "train_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
        "val_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
        "output_dir": "./runs/autoresearch_5090/promoted_current",
        "run_name": None,
        "results_tsv_path": "./runs/autoresearch_5090/results.tsv",
        "metrics_jsonl_path": None,
        "tensorboard_log_dir": None,
        "resume_from": None,
        "load_artifact_path": None,
        "max_wallclock_seconds": 300.0,
    }


def portable_h100_defaults(mode: str) -> dict[str, Any]:
    if mode == "8x":
        return {
            "train_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
            "val_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
            "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
            "output_dir": "./runs/runpod_h100_8x_10min/promoted",
            "run_name": None,
            "results_tsv_path": "./runs/runpod_h100_8x_10min/results.tsv",
            "metrics_jsonl_path": None,
            "tensorboard_log_dir": None,
            "resume_from": None,
            "load_artifact_path": None,
            "max_wallclock_seconds": 600.0,
            "log_every": 20,
            "val_every": 100,
            "checkpoint_every": 100,
            "use_lawa": True,
            "verify_export_reload": True,
        }
    if mode == "1x":
        return {
            "train_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
            "val_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
            "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
            "output_dir": "./runs/runpod_h100_1x_10min/promoted",
            "run_name": None,
            "results_tsv_path": "./runs/runpod_h100_1x_10min/results.tsv",
            "metrics_jsonl_path": None,
            "tensorboard_log_dir": None,
            "resume_from": None,
            "load_artifact_path": None,
            "max_wallclock_seconds": 600.0,
            "log_every": 20,
            "val_every": 100,
            "checkpoint_every": 100,
            "use_lawa": True,
            "verify_export_reload": True,
        }
    raise ValueError(f"unsupported promoted h100 mode: {mode}")


def portable_promoted_config(source_config: dict[str, Any], target: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_config))
    defaults = portable_5090_defaults() if target == "5090" else portable_h100_defaults(target)
    for key, value in defaults.items():
        payload[key] = value
    return payload


def load_tracked_accepted_state(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != TRACKED_ACCEPTED_STATE_SCHEMA_VERSION:
        raise ValueError(f"unexpected tracked accepted state schema: {payload.get('schema_version')}")
    return payload


def sync_tracked_accepted_state(
    session: Mapping[str, Any],
    *,
    resolved_results: Path | None = None,
    results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted_payload: dict[str, Any] = {
        "schema_version": TRACKED_ACCEPTED_STATE_SCHEMA_VERSION,
        "updated_at_unix": time.time(),
        "current_branch": current_branch(),
        "current_commit": current_commit(),
        "current_commit_short": current_commit_short(),
        "accepted_commit": session.get("accepted_commit"),
        "accepted_commit_short": session.get("accepted_commit_short"),
        "accepted_run_id": session.get("accepted_run_id"),
        "accepted_val_bpb": session.get("accepted_val_bpb"),
        "accepted_artifact_bytes": session.get("accepted_artifact_bytes"),
        "accepted_results_path": None if resolved_results is None else str(resolved_results),
        "search_policy": dict(session.get("search_policy") or default_search_policy()),
    }

    if results is not None and results.get("config_path"):
        source_config_path = Path(str(results["config_path"]))
        if source_config_path.is_file():
            source_config = load_json(source_config_path)
            promoted_5090_path = tracked_promoted_5090_config_path()
            promoted_h100_8x_path = tracked_promoted_h100_config_path("8x")
            promoted_h100_1x_path = tracked_promoted_h100_config_path("1x")
            atomic_write_json(promoted_5090_path, portable_promoted_config(source_config, "5090"))
            atomic_write_json(promoted_h100_8x_path, portable_promoted_config(source_config, "8x"))
            atomic_write_json(promoted_h100_1x_path, portable_promoted_config(source_config, "1x"))
            accepted_payload["source_config_path"] = str(source_config_path)
            accepted_payload["promoted_5090_config_path"] = repo_relative_path(promoted_5090_path)
            accepted_payload["promoted_h100_8x_config_path"] = repo_relative_path(promoted_h100_8x_path)
            accepted_payload["promoted_h100_1x_config_path"] = repo_relative_path(promoted_h100_1x_path)

    atomic_write_json(tracked_accepted_state_path(), accepted_payload)
    return accepted_payload


def ensure_notes_file(state_dir: Path) -> None:
    path = notes_path(state_dir)
    if path.exists():
        return
    path.write_text(
        "# Autoresearch Notes\n\n"
        "Accepted state:\n"
        "- The accepted code state is the current branch tip plus `state/autoresearch/accepted_state.json`.\n"
        "- The promoted cross-host configs live under `configs/promoted/`.\n"
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
            **default_search_policy(),
        },
    }
    write_session(state_dir, session)
    ensure_notes_file(state_dir)
    ensure_loop_support_files(state_dir)
    sync_tracked_accepted_state(session, resolved_results=resolved_results, results=baseline)
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


def init_session_from_tracked(state_dir: Path, tracked_state_path_value: Path, force: bool = False) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = session_path(state_dir)
    if path.exists() and not force:
        raise ValueError(f"autoresearch session already exists: {path}")

    tracked = load_tracked_accepted_state(tracked_state_path_value)
    branch = current_branch()
    commit = current_commit()
    commit_short = current_commit_short()
    now = time.time()
    accepted_results_path = tracked.get("accepted_results_path") or str(tracked_state_path_value)
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "ready",
        "created_at_unix": now,
        "updated_at_unix": now,
        "repo_root": str(repo_root()),
        "state_dir": str(state_dir),
        "current_branch": branch,
        "accepted_commit": tracked.get("accepted_commit") or commit,
        "accepted_commit_short": tracked.get("accepted_commit_short") or commit_short,
        "accepted_run_id": tracked["accepted_run_id"],
        "accepted_val_bpb": tracked["accepted_val_bpb"],
        "accepted_artifact_bytes": tracked["accepted_artifact_bytes"],
        "accepted_results_path": accepted_results_path,
        "baseline_run_id": tracked["accepted_run_id"],
        "baseline_val_bpb": tracked["accepted_val_bpb"],
        "baseline_artifact_bytes": tracked["accepted_artifact_bytes"],
        "baseline_results_path": accepted_results_path,
        "latest_run_id": tracked["accepted_run_id"],
        "latest_results_path": accepted_results_path,
        "latest_status": "success",
        "latest_val_bpb": tracked["accepted_val_bpb"],
        "latest_artifact_bytes": tracked["accepted_artifact_bytes"],
        "latest_decision": "accepted",
        "current_experiment": None,
        "search_policy": tracked.get("search_policy") or default_search_policy(),
        "tracked_accepted_state_path": str(tracked_state_path_value),
    }
    write_session(state_dir, session)
    ensure_notes_file(state_dir)
    ensure_loop_support_files(state_dir)
    append_experiment_event(
        state_dir,
        {
            "event": "tracked_init",
            "run_id": tracked["accepted_run_id"],
            "results_path": accepted_results_path,
            "branch": branch,
            "commit": commit,
            "commit_short": commit_short,
            "status": "success",
            "val_bpb": tracked["accepted_val_bpb"],
            "artifact_bytes": tracked["accepted_artifact_bytes"],
            "decision": "accepted",
            "tracked_state_path": str(tracked_state_path_value),
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
    if decision == "accepted":
        sync_tracked_accepted_state(session, resolved_results=resolved_results, results=results)
    return load_session(state_dir)


def sync_current_tracked_accepted_state(state_dir: Path, results_json: Path | None = None) -> dict[str, Any]:
    session = load_session(state_dir)
    resolved_results: Path | None = None
    results: dict[str, Any] | None = None
    if results_json is not None:
        resolved_results = resolve_results_source(results_json)
    else:
        accepted_results_path = session.get("accepted_results_path")
        if accepted_results_path:
            candidate = Path(str(accepted_results_path))
            if candidate.is_file():
                resolved_results = resolve_results_source(candidate)
    if resolved_results is not None and resolved_results.is_file():
        results = dict(load_and_validate_results(resolved_results))
    return sync_tracked_accepted_state(session, resolved_results=resolved_results, results=results)


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Ralph-style autoresearch session state.")
    parser.add_argument("--state_dir", type=str, default=str(default_state_dir()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--baseline_results", type=str, required=True)
    init_parser.add_argument("--force", action="store_true")

    tracked_init_parser = subparsers.add_parser("init-from-tracked")
    tracked_init_parser.add_argument("--tracked_state", type=str, required=True)
    tracked_init_parser.add_argument("--force", action="store_true")

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

    sync_parser = subparsers.add_parser("sync-tracked-accepted")
    sync_parser.add_argument("--results_json", type=str, default=None)

    args = parser.parse_args()
    state_dir = Path(args.state_dir)

    if args.command == "init":
        print_payload(init_session(state_dir, Path(args.baseline_results), force=args.force))
        return
    if args.command == "init-from-tracked":
        print_payload(init_session_from_tracked(state_dir, Path(args.tracked_state), force=args.force))
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
    if args.command == "sync-tracked-accepted":
        print_payload(sync_current_tracked_accepted_state(state_dir, None if args.results_json is None else Path(args.results_json)))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
