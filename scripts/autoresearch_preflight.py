from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from train import atomic_write_json, load_and_validate_results


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_session(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "session.json"
    if not path.is_file():
        raise ValueError(f"missing session.json: {path}")
    return load_json(path)


def resolve_results_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def get_benchmark(results: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if results is None:
        return None
    bench = results.get("benchmark")
    return bench if isinstance(bench, Mapping) else None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def decide_preflight(session: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    search_policy = dict(session.get("search_policy") or {})
    preflight_policy = dict(search_policy.get("preflight") or {})
    enabled = bool(preflight_policy.get("enabled", False))
    baseline_results: Mapping[str, Any] | None = None
    baseline_path = resolve_results_path(session.get("accepted_results_path") or session.get("baseline_results_path"))
    if baseline_path is not None and baseline_path.is_file():
        try:
            baseline_results = dict(load_and_validate_results(baseline_path))
        except Exception:
            baseline_results = None

    baseline_bench = get_benchmark(baseline_results)
    candidate_bench = get_benchmark(candidate)

    candidate_tps = None if candidate_bench is None else safe_float(candidate_bench.get("train_tokens_per_second"))
    baseline_tps = None if baseline_bench is None else safe_float(baseline_bench.get("train_tokens_per_second"))
    train_tps_ratio = None
    if candidate_tps is not None and baseline_tps not in (None, 0.0):
        train_tps_ratio = candidate_tps / baseline_tps

    candidate_total_tokens = safe_int(candidate.get("total_tokens"))
    baseline_total_tokens = safe_int(None if baseline_results is None else baseline_results.get("total_tokens"))
    total_tokens_ratio = None
    if candidate_total_tokens is not None and baseline_total_tokens not in (None, 0):
        total_tokens_ratio = candidate_total_tokens / baseline_total_tokens

    candidate_artifact = safe_int(candidate.get("artifact_bytes"))
    baseline_artifact = safe_int(session.get("accepted_artifact_bytes"))
    artifact_gain_bytes = None
    if candidate_artifact is not None and baseline_artifact is not None:
        artifact_gain_bytes = candidate_artifact - baseline_artifact

    hard_artifact_max = int(preflight_policy.get("hard_artifact_bytes_max", 16_000_000))
    soft_artifact_max = safe_int(preflight_policy.get("soft_artifact_bytes_max"))
    soft_artifact_min = safe_int(preflight_policy.get("soft_artifact_bytes_min"))
    min_train_ratio = safe_float(preflight_policy.get("min_train_tokens_per_second_ratio"))
    min_total_ratio = safe_float(preflight_policy.get("min_total_tokens_ratio"))
    allow_slow_if_within_soft_cap = bool(preflight_policy.get("allow_slow_if_within_soft_cap", True))
    allow_slow_if_artifact_gain_bytes = safe_int(preflight_policy.get("allow_slow_if_artifact_gain_bytes")) or 0

    reasons: list[str] = []
    warnings: list[str] = []

    if candidate.get("status") != "success":
        reasons.append("benchmark_failed")
    if candidate.get("mode") != "benchmark":
        warnings.append(f"unexpected_mode={candidate.get('mode')}")

    if candidate_artifact is None:
        warnings.append("missing_artifact_bytes")
    else:
        if candidate_artifact > hard_artifact_max:
            reasons.append("artifact_over_hard_cap")
        elif soft_artifact_max is not None and candidate_artifact > soft_artifact_max:
            warnings.append("artifact_over_soft_cap")
        if soft_artifact_min is not None and candidate_artifact < soft_artifact_min:
            warnings.append("artifact_under_soft_band")

    slow_train = min_train_ratio is not None and train_tps_ratio is not None and train_tps_ratio < min_train_ratio
    slow_tokens = min_total_ratio is not None and total_tokens_ratio is not None and total_tokens_ratio < min_total_ratio
    enough_artifact_gain = artifact_gain_bytes is not None and artifact_gain_bytes >= allow_slow_if_artifact_gain_bytes
    within_soft_cap = candidate_artifact is not None and (soft_artifact_max is None or candidate_artifact <= soft_artifact_max)

    if slow_train:
        warnings.append("train_throughput_below_gate")
    if slow_tokens:
        warnings.append("projected_total_tokens_below_gate")

    if slow_train or slow_tokens:
        if not ((allow_slow_if_within_soft_cap and within_soft_cap) or enough_artifact_gain):
            reasons.append("throughput_gate_failed")

    decision = "proceed" if enabled and not reasons else ("skip" if enabled else "disabled")
    if enabled and not reasons and slow_train:
        warnings.append("slow_but_allowed")
    if enabled and not reasons and slow_tokens:
        warnings.append("token_budget_low_but_allowed")

    return {
        "enabled": enabled,
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "lane": search_policy.get("lane"),
        "policy": preflight_policy,
        "baseline": {
            "accepted_run_id": session.get("accepted_run_id"),
            "accepted_val_bpb": session.get("accepted_val_bpb"),
            "accepted_artifact_bytes": baseline_artifact,
            "train_tokens_per_second": baseline_tps,
            "total_tokens": baseline_total_tokens,
        },
        "candidate": {
            "run_id": candidate.get("run_id"),
            "val_bpb": candidate.get("val_bpb"),
            "artifact_bytes": candidate_artifact,
            "train_tokens_per_second": candidate_tps,
            "total_tokens": candidate_total_tokens,
        },
        "ratios": {
            "train_tokens_per_second": train_tps_ratio,
            "total_tokens": total_tokens_ratio,
        },
        "artifact_gain_bytes": artifact_gain_bytes,
    }


def attach_preflight(results_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    results = dict(load_and_validate_results(results_path))
    results["preflight"] = dict(payload)
    atomic_write_json(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark-mode preflight for autoresearch runs.")
    parser.add_argument("--state_dir", type=str, required=True)
    parser.add_argument("--results_json", type=str, required=True)
    parser.add_argument("--write_json", type=str, default=None)
    parser.add_argument("--attach_to_results", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    results_path = Path(args.results_json)
    session = load_session(state_dir)
    candidate = dict(load_and_validate_results(results_path))
    payload = decide_preflight(session, candidate)
    if args.attach_to_results:
        attach_preflight(results_path, payload)
    if args.write_json:
        atomic_write_json(Path(args.write_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
