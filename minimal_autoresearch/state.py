from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


STATE_SCHEMA_VERSION = "pgolf.minimal_autoresearch_state.v1"
ATTEMPT_SCHEMA_VERSION = "pgolf.minimal_autoresearch_attempt.v1"
DEFAULT_CONFIG_JSON = "configs/autoresearch_5090_frontier_5min.json"
DEFAULT_MIN_IMPROVEMENT = 0.0005
DEFAULT_ARTIFACT_HARD_MAX = 16_000_000
DEFAULT_TRAINING_SECONDS_MIN_RATIO = 0.70
DEFAULT_TRAINING_SECONDS_MAX_RATIO = 1.40
PROTECTED_CONFIG_FIELDS = (
    "max_wallclock_seconds",
    "iterations",
    "train_pattern",
    "val_pattern",
    "tokenizer_path",
    "train_batch_tokens",
    "val_batch_tokens",
    "grad_accum_steps",
    "seed",
    "deterministic",
    "val_every",
    "eval_first_step",
    "benchmark_only",
    "benchmark_train_steps",
    "benchmark_eval_repeats",
    "checkpoint_every",
    "use_compile",
    "use_lawa",
    "save_final_quantized",
    "verify_export_reload",
    "counted_code_paths",
    "resume_from",
    "load_artifact_path",
    "evaluate_only",
    "train_phase_only",
)
PROTECTED_PATH_FIELDS = frozenset(
    {
        "train_pattern",
        "val_pattern",
        "tokenizer_path",
        "resume_from",
        "load_artifact_path",
    }
)
BOOLEAN_PROTOCOL_FIELDS = frozenset(
    {
        "deterministic",
        "eval_first_step",
        "benchmark_only",
        "use_compile",
        "use_lawa",
        "save_final_quantized",
        "verify_export_reload",
        "evaluate_only",
        "train_phase_only",
    }
)


def repo_root() -> Path:
    return ROOT_DIR


def default_state_dir() -> Path:
    return repo_root() / ".minimal_autoresearch"


def state_path(state_dir: Path) -> Path:
    return state_dir / "state.json"


def attempts_path(state_dir: Path) -> Path:
    return state_dir / "attempts.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root() / path


def repo_relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def resolve_results_source(path_value: str | Path) -> Path:
    path = resolve_repo_path(path_value)
    payload = load_json(path)
    indexed_source = payload.get("indexed_source_results_path")
    if indexed_source:
        return resolve_repo_path(str(indexed_source))
    return path


def normalize_repoish_path(value: str | Path) -> str:
    text = str(value)
    if text.startswith("./"):
        text = text[2:]
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(repo_root()))
        except ValueError:
            marker = f"/{repo_root().name}/"
            if marker in text:
                return text.split(marker, 1)[1]
    return text


def normalize_protocol_value(field: str, value: Any) -> Any:
    if field in BOOLEAN_PROTOCOL_FIELDS and value is None:
        return False
    if field in PROTECTED_PATH_FIELDS:
        if value is None:
            return None
        return normalize_repoish_path(value)
    if field == "counted_code_paths":
        if value is None:
            return None
        return [normalize_repoish_path(item) for item in value]
    return value


def build_protocol_from_config_payload(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: normalize_protocol_value(field, config_payload.get(field))
        for field in PROTECTED_CONFIG_FIELDS
    }


def load_protocol_from_config_json(config_json: str | Path) -> tuple[dict[str, Any], Path]:
    resolved_config = resolve_repo_path(config_json)
    if not resolved_config.is_file():
        raise ValueError(f"missing config json: {resolved_config}")
    return build_protocol_from_config_payload(load_json(resolved_config)), resolved_config


def protocol_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    protocol = state.get("protocol")
    if isinstance(protocol, Mapping):
        return {
            field: normalize_protocol_value(field, protocol.get(field))
            for field in PROTECTED_CONFIG_FIELDS
        }
    config_json = state.get("config_json")
    if not config_json:
        raise ValueError("minimal autoresearch state missing protocol and config_json")
    protocol, _resolved = load_protocol_from_config_json(str(config_json))
    return protocol


def load_and_validate_results(path_value: str | Path) -> Mapping[str, Any]:
    resolved_path = resolve_repo_path(path_value)
    payload = load_json(resolved_path)
    required = {
        "schema_version": str,
        "status": str,
        "mode": str,
        "run_id": str,
        "output_dir": str,
        "results_path": str,
        "training_seconds": (int, float),
        "total_seconds": (int, float),
        "artifact_bytes": int,
        "val_bpb": (int, float, type(None)),
    }
    for key, expected in required.items():
        if key not in payload:
            raise ValueError(f"results.json missing required key: {key}")
        if not isinstance(payload[key], expected):
            raise ValueError(f"results.json key {key!r} has invalid type: {type(payload[key]).__name__}")
    payload["_resolved_results_source_path"] = str(resolved_path)
    return payload


def load_results_config(results: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    config_path_value = results.get("config_path")
    source_path_value = results.get("_resolved_results_source_path")
    if not config_path_value:
        if source_path_value:
            sibling_path = Path(str(source_path_value)).parent / "config.json"
            if sibling_path.is_file():
                return load_json(sibling_path), sibling_path
        return None, None
    config_path = resolve_repo_path(str(config_path_value))
    if not config_path.is_file() and source_path_value:
        sibling_path = Path(str(source_path_value)).parent / "config.json"
        if sibling_path.is_file():
            config_path = sibling_path
    if not config_path.is_file():
        return None, config_path
    try:
        return load_json(config_path), config_path
    except Exception:
        return None, config_path


def load_train_loop_seconds(results: Mapping[str, Any]) -> float | None:
    metrics_path_value = results.get("metrics_path")
    source_path_value = results.get("_resolved_results_source_path")
    metrics_path: Path | None = None
    if metrics_path_value:
        metrics_path = resolve_repo_path(str(metrics_path_value))
    if (metrics_path is None or not metrics_path.is_file()) and source_path_value:
        sibling_path = Path(str(source_path_value)).parent / "metrics.jsonl"
        if sibling_path.is_file():
            metrics_path = sibling_path
    if metrics_path is None or not metrics_path.is_file():
        return None
    last_elapsed: float | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if payload.get("event") != "train":
                continue
            elapsed = payload.get("elapsed_training_seconds")
            if elapsed is not None:
                last_elapsed = float(elapsed)
    return last_elapsed


def git_output(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=repo_root(),
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


def load_state(state_dir: Path) -> dict[str, Any]:
    path = state_path(state_dir)
    if not path.is_file():
        raise ValueError(f"missing minimal autoresearch state: {path}")
    payload = load_json(path)
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"unexpected minimal autoresearch state schema: {payload.get('schema_version')}")
    return payload


def write_state(state_dir: Path, payload: Mapping[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    output = dict(payload)
    output["schema_version"] = STATE_SCHEMA_VERSION
    output["updated_at_unix"] = time.time()
    return atomic_write_json(state_path(state_dir), output)


def ensure_support_files(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = attempts_path(state_dir)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def load_recent_attempts(state_dir: Path, limit: int) -> list[dict[str, Any]]:
    path = attempts_path(state_dir)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recent = lines[-max(0, limit) :]
    return [json.loads(line) for line in recent]


def resolve_expected_training_seconds(state: Mapping[str, Any]) -> float | None:
    value = protocol_from_state(state).get("max_wallclock_seconds")
    if value is None:
        return None
    return float(value)


def validate_results_protocol(state: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    expected_protocol = protocol_from_state(state)
    results_config, config_path = load_results_config(results)
    if results_config is None:
        return {
            "valid": False,
            "missing_config": True,
            "run_config_path": None if config_path is None else repo_relative_path(config_path),
            "mismatches": [],
        }

    mismatches: list[dict[str, Any]] = []
    for field, expected_value in expected_protocol.items():
        actual_value = normalize_protocol_value(field, results_config.get(field))
        if actual_value != expected_value:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return {
        "valid": not mismatches,
        "missing_config": False,
        "run_config_path": None if config_path is None else repo_relative_path(config_path),
        "mismatches": mismatches,
    }


def assess_results(state: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []

    accepted_val_bpb = state.get("accepted_val_bpb")
    candidate_val_bpb = results.get("val_bpb")
    artifact_bytes = results.get("artifact_bytes")
    hard_cap = int(state.get("artifact_bytes_hard_max", DEFAULT_ARTIFACT_HARD_MAX))
    min_improvement = float(state.get("min_improvement", DEFAULT_MIN_IMPROVEMENT))
    protocol_check = validate_results_protocol(state, results)
    expected_training_seconds = resolve_expected_training_seconds(state)
    training_seconds = results.get("training_seconds")
    train_loop_seconds = load_train_loop_seconds(results)
    measured_training_seconds = train_loop_seconds
    measured_training_seconds_source = "metrics.train.elapsed_training_seconds"
    if measured_training_seconds is None:
        measured_training_seconds = training_seconds
        measured_training_seconds_source = "results.training_seconds"
        if training_seconds is None:
            warnings.append("missing_training_seconds")
        else:
            warnings.append("missing_train_loop_seconds")
            warnings.append("using_summary_training_seconds")
    training_seconds_ratio = None
    training_budget_valid = True
    if expected_training_seconds not in (None, 0.0) and measured_training_seconds is not None:
        training_seconds_ratio = float(measured_training_seconds) / float(expected_training_seconds)
        training_budget_valid = (
            float(state.get("training_seconds_min_ratio", DEFAULT_TRAINING_SECONDS_MIN_RATIO))
            <= training_seconds_ratio
            <= float(state.get("training_seconds_max_ratio", DEFAULT_TRAINING_SECONDS_MAX_RATIO))
        )

    artifact_valid = artifact_bytes is not None and int(artifact_bytes) <= hard_cap
    if not artifact_valid:
        reasons.append("artifact_over_hard_cap")
    if results.get("status") != "success":
        reasons.append("status_not_success")
    if results.get("mode") != "train":
        reasons.append("mode_not_train")
    if candidate_val_bpb is None:
        reasons.append("missing_val_bpb")
    if protocol_check["missing_config"]:
        reasons.append("missing_run_config")
    elif not protocol_check["valid"]:
        reasons.append("protocol_drift")
    if not training_budget_valid:
        reasons.append("training_budget_out_of_range")

    improvement = None
    if accepted_val_bpb is not None and candidate_val_bpb is not None:
        improvement = float(accepted_val_bpb) - float(candidate_val_bpb)
        if improvement < min_improvement:
            reasons.append("not_better_than_current_best")

    decision = "accept" if not reasons else "reject"
    return {
        "decision": decision,
        "accepted_run_id": state.get("accepted_run_id"),
        "accepted_val_bpb": accepted_val_bpb,
        "candidate_run_id": results.get("run_id"),
        "candidate_val_bpb": candidate_val_bpb,
        "improvement": improvement,
        "min_improvement": min_improvement,
        "artifact_bytes": artifact_bytes,
        "artifact_bytes_hard_max": hard_cap,
        "artifact_valid": artifact_valid,
        "expected_training_seconds": expected_training_seconds,
        "training_seconds": training_seconds,
        "train_loop_seconds": train_loop_seconds,
        "measured_training_seconds": measured_training_seconds,
        "measured_training_seconds_source": measured_training_seconds_source,
        "training_seconds_ratio": training_seconds_ratio,
        "training_budget_valid": training_budget_valid,
        "protocol_valid": protocol_check["valid"],
        "protocol_missing_config": protocol_check["missing_config"],
        "protocol_run_config_path": protocol_check["run_config_path"],
        "protocol_mismatches": protocol_check["mismatches"],
        "reasons": reasons,
        "warnings": warnings,
    }


def init_state(
    state_dir: Path,
    baseline_results_path_value: Path,
    *,
    config_json: str,
    force: bool,
) -> dict[str, Any]:
    if state_path(state_dir).exists() and not force:
        raise ValueError(f"minimal autoresearch state already exists: {state_path(state_dir)}")

    resolved_results = resolve_results_source(baseline_results_path_value)
    baseline = dict(load_and_validate_results(resolved_results))
    if baseline.get("status") != "success" or baseline.get("mode") != "train" or baseline.get("val_bpb") is None:
        raise ValueError("baseline results must be a successful train run with val_bpb")
    protocol, resolved_config_json = load_protocol_from_config_json(config_json)
    baseline_protocol_check = validate_results_protocol(
        {"config_json": repo_relative_path(resolved_config_json), "protocol": protocol},
        baseline,
    )
    if baseline_protocol_check["missing_config"] or not baseline_protocol_check["valid"]:
        raise ValueError(f"baseline results do not match frozen protocol: {baseline_protocol_check}")

    now = time.time()
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at_unix": now,
        "updated_at_unix": now,
        "current_branch": current_branch(),
        "accepted_commit": current_commit(),
        "accepted_commit_short": current_commit_short(),
        "accepted_run_id": baseline["run_id"],
        "accepted_results_path": repo_relative_path(resolved_results),
        "accepted_val_bpb": baseline["val_bpb"],
        "accepted_artifact_bytes": baseline["artifact_bytes"],
        "baseline_run_id": baseline["run_id"],
        "baseline_results_path": repo_relative_path(resolved_results),
        "baseline_val_bpb": baseline["val_bpb"],
        "baseline_artifact_bytes": baseline["artifact_bytes"],
        "latest_run_id": baseline["run_id"],
        "latest_results_path": repo_relative_path(resolved_results),
        "latest_status": baseline["status"],
        "latest_val_bpb": baseline["val_bpb"],
        "latest_decision": "accepted",
        "config_json": repo_relative_path(resolved_config_json),
        "protocol": protocol,
        "min_improvement": DEFAULT_MIN_IMPROVEMENT,
        "artifact_bytes_hard_max": DEFAULT_ARTIFACT_HARD_MAX,
        "training_seconds_min_ratio": DEFAULT_TRAINING_SECONDS_MIN_RATIO,
        "training_seconds_max_ratio": DEFAULT_TRAINING_SECONDS_MAX_RATIO,
    }
    ensure_support_files(state_dir)
    write_state(state_dir, payload)
    return load_state(state_dir)


def show_state(state_dir: Path, recent: int) -> dict[str, Any]:
    payload = load_state(state_dir)
    output = dict(payload)
    output["protocol"] = protocol_from_state(payload)
    output["recent_attempts"] = load_recent_attempts(state_dir, recent)
    return output


def record_attempt(
    state_dir: Path,
    *,
    decision: str,
    results_json: Path | None,
    run_id: str | None,
    experiment_commit: str | None,
    revert_commit: str | None,
    notes: str | None,
) -> dict[str, Any]:
    if decision not in {"accepted", "reverted"}:
        raise ValueError(f"unsupported decision: {decision}")

    ensure_support_files(state_dir)
    state = load_state(state_dir)
    results: dict[str, Any] | None = None
    resolved_results: Path | None = None
    assessment: dict[str, Any] | None = None
    if results_json is not None:
        resolved_results = resolve_results_source(results_json)
        results = dict(load_and_validate_results(resolved_results))
        assessment = assess_results(state, results)
    if decision == "accepted":
        if results is None or assessment is None:
            raise ValueError("accepted attempts require --results_json")
        if assessment.get("decision") != "accept":
            raise ValueError(f"cannot record accepted attempt: {assessment}")
    final_run_id = run_id or (None if results is None else str(results["run_id"]))
    if final_run_id is None:
        raise ValueError("record requires --run_id or --results_json")

    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "recorded_at_unix": time.time(),
        "decision": decision,
        "run_id": final_run_id,
        "results_json": repo_relative_path(resolved_results),
        "status": None if results is None else results.get("status"),
        "mode": None if results is None else results.get("mode"),
        "val_bpb": None if results is None else results.get("val_bpb"),
        "artifact_bytes": None if results is None else results.get("artifact_bytes"),
        "training_seconds": None if results is None else results.get("training_seconds"),
        "train_loop_seconds": None if assessment is None else assessment.get("train_loop_seconds"),
        "experiment_commit": experiment_commit,
        "revert_commit": revert_commit,
        "notes": notes,
        "assessment": assessment,
    }
    append_jsonl(attempts_path(state_dir), attempt)

    state = dict(state)
    state["current_branch"] = current_branch()
    state["latest_run_id"] = final_run_id
    state["latest_results_path"] = repo_relative_path(resolved_results)
    state["latest_status"] = None if results is None else results.get("status")
    state["latest_val_bpb"] = None if results is None else results.get("val_bpb")
    state["latest_decision"] = decision
    if decision == "accepted" and results is not None:
        state["accepted_commit"] = current_commit()
        state["accepted_commit_short"] = current_commit_short()
        state["accepted_run_id"] = str(results["run_id"])
        state["accepted_results_path"] = repo_relative_path(resolved_results)
        state["accepted_val_bpb"] = results["val_bpb"]
        state["accepted_artifact_bytes"] = results["artifact_bytes"]
    write_state(state_dir, state)
    return show_state(state_dir, recent=5)


def print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal accepted-state helper for Codex autoresearch.")
    parser.add_argument("--state_dir", type=str, default=str(default_state_dir()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--baseline_results", type=str, required=True)
    init_parser.add_argument("--config_json", type=str, default=DEFAULT_CONFIG_JSON)
    init_parser.add_argument("--force", action="store_true")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--recent", type=int, default=5)

    assess_parser = subparsers.add_parser("assess")
    assess_parser.add_argument("--results_json", type=str, required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--decision", type=str, choices=("accepted", "reverted"), required=True)
    record_parser.add_argument("--results_json", type=str, default=None)
    record_parser.add_argument("--run_id", type=str, default=None)
    record_parser.add_argument("--experiment_commit", type=str, default=None)
    record_parser.add_argument("--revert_commit", type=str, default=None)
    record_parser.add_argument("--notes", type=str, default=None)

    args = parser.parse_args()
    state_dir = Path(args.state_dir)

    if args.command == "init":
        print_payload(
            init_state(
                state_dir,
                Path(args.baseline_results),
                config_json=args.config_json,
                force=args.force,
            )
        )
        return
    if args.command == "show":
        print_payload(show_state(state_dir, recent=args.recent))
        return
    if args.command == "assess":
        state = load_state(state_dir)
        results = dict(load_and_validate_results(resolve_results_source(Path(args.results_json))))
        print_payload(assess_results(state, results))
        return
    if args.command == "record":
        print_payload(
            record_attempt(
                state_dir,
                decision=args.decision,
                results_json=None if args.results_json is None else Path(args.results_json),
                run_id=args.run_id,
                experiment_commit=args.experiment_commit,
                revert_commit=args.revert_commit,
                notes=args.notes,
            )
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
