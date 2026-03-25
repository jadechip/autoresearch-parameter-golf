from __future__ import annotations

import argparse
import json
import math
from typing import Any, Mapping

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from train import atomic_write_json, load_and_validate_results


DEFAULT_PROJECTED_ARTIFACT_MARGIN_BYTES = 250_000


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


def load_results_if_present(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return dict(load_and_validate_results(path))
    except Exception:
        return None


def expected_training_budget_seconds(results: Mapping[str, Any] | None) -> float | None:
    if results is None:
        return None
    config_path_value = results.get("config_path")
    if not config_path_value:
        return None
    config_path = Path(str(config_path_value))
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    if not config_path.is_file():
        return None
    try:
        config_payload = load_json(config_path)
    except Exception:
        return None
    return safe_float(config_payload.get("max_wallclock_seconds"))


def derive_train_tokens_per_second(results: Mapping[str, Any] | None) -> float | None:
    if results is None:
        return None
    bench = get_benchmark(results)
    if bench is not None:
        value = safe_float(bench.get("train_tokens_per_second"))
        if value is not None and value > 0:
            return value
    total_tokens = safe_float(results.get("total_tokens"))
    training_seconds = safe_float(results.get("training_seconds"))
    if total_tokens is None or training_seconds in (None, 0.0):
        return None
    return total_tokens / training_seconds


def derive_total_tokens(results: Mapping[str, Any] | None) -> int | None:
    if results is None:
        return None
    return safe_int(results.get("total_tokens"))


def resolve_related_preflight_results(train_results_path: Path | None, run_id: str | None) -> Path | None:
    if train_results_path is None or not run_id:
        return None
    run_dir = train_results_path.parent
    runs_root = run_dir.parent
    candidate = runs_root / f"{run_id}-preflight" / "results.json"
    return candidate if candidate.is_file() else None


def artifact_stats(results: Mapping[str, Any] | None) -> dict[str, int | None]:
    if results is None:
        return {
            "artifact_bytes": None,
            "code_bytes": None,
            "compressed_model_bytes": None,
            "quant_payload_bytes": None,
        }
    payload = results.get("artifact")
    artifact_payload = payload if isinstance(payload, Mapping) else {}
    artifact_bytes = safe_int(results.get("artifact_bytes"))
    code_bytes = safe_int(artifact_payload.get("code_bytes"))
    compressed_model_bytes = safe_int(artifact_payload.get("compressed_model_bytes"))
    quant_payload_bytes = safe_int(artifact_payload.get("quant_payload_bytes"))
    if compressed_model_bytes is None and artifact_bytes is not None and code_bytes is not None:
        compressed_model_bytes = max(0, artifact_bytes - code_bytes)
    return {
        "artifact_bytes": artifact_bytes,
        "code_bytes": code_bytes,
        "compressed_model_bytes": compressed_model_bytes,
        "quant_payload_bytes": quant_payload_bytes,
    }


def compression_ratio(artifact: Mapping[str, int | None]) -> float | None:
    quant_payload_bytes = artifact.get("quant_payload_bytes")
    compressed_model_bytes = artifact.get("compressed_model_bytes")
    if quant_payload_bytes in (None, 0) or compressed_model_bytes is None:
        return None
    return float(compressed_model_bytes) / float(quant_payload_bytes)


def choose_reference_compression_ratio(
    baseline_train_artifact: Mapping[str, int | None],
    baseline_preflight_artifact: Mapping[str, int | None],
    candidate_artifact: Mapping[str, int | None],
) -> tuple[float | None, str | None]:
    choices: list[tuple[str, float]] = []
    for label, artifact in (
        ("baseline_train", baseline_train_artifact),
        ("baseline_preflight", baseline_preflight_artifact),
        ("candidate_preflight", candidate_artifact),
    ):
        ratio = compression_ratio(artifact)
        if ratio is not None:
            choices.append((label, ratio))
    if not choices:
        return None, None
    label, ratio = max(choices, key=lambda item: item[1])
    return ratio, label


def project_artifact_bytes(
    *,
    baseline_train_artifact: Mapping[str, int | None],
    baseline_preflight_artifact: Mapping[str, int | None],
    candidate_artifact: Mapping[str, int | None],
    margin_bytes: int,
) -> dict[str, Any]:
    reference_ratio, reference_source = choose_reference_compression_ratio(
        baseline_train_artifact,
        baseline_preflight_artifact,
        candidate_artifact,
    )
    code_bytes = candidate_artifact.get("code_bytes") or 0
    quant_payload_bytes = candidate_artifact.get("quant_payload_bytes")
    observed_artifact_bytes = candidate_artifact.get("artifact_bytes")
    observed_model_bytes = candidate_artifact.get("compressed_model_bytes") or 0
    projected_model_bytes = observed_model_bytes
    if quant_payload_bytes is not None and reference_ratio is not None:
        projected_model_bytes = max(projected_model_bytes, int(math.ceil(reference_ratio * float(quant_payload_bytes))))
    projected_artifact_bytes = code_bytes + projected_model_bytes + max(0, int(margin_bytes))
    if observed_artifact_bytes is not None:
        projected_artifact_bytes = max(projected_artifact_bytes, int(observed_artifact_bytes))
    return {
        "reference_compression_ratio": reference_ratio,
        "reference_source": reference_source,
        "margin_bytes": int(margin_bytes),
        "observed_artifact_bytes": observed_artifact_bytes,
        "projected_artifact_bytes": projected_artifact_bytes,
        "code_bytes": candidate_artifact.get("code_bytes"),
        "compressed_model_bytes": candidate_artifact.get("compressed_model_bytes"),
        "quant_payload_bytes": candidate_artifact.get("quant_payload_bytes"),
    }


def decide_preflight(session: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    search_policy = dict(session.get("search_policy") or {})
    preflight_policy = dict(search_policy.get("preflight") or {})
    enabled = bool(preflight_policy.get("enabled", False))

    baseline_train_path = resolve_results_path(session.get("accepted_results_path") or session.get("baseline_results_path"))
    baseline_train_results = load_results_if_present(baseline_train_path)
    baseline_preflight_path = resolve_related_preflight_results(
        baseline_train_path,
        None if baseline_train_results is None else str(baseline_train_results.get("run_id") or ""),
    )
    baseline_preflight_results = load_results_if_present(baseline_preflight_path)

    candidate_tps = derive_train_tokens_per_second(candidate)
    baseline_tps_source = "train_results"
    baseline_tps = derive_train_tokens_per_second(baseline_train_results)
    if baseline_tps is None:
        baseline_tps = derive_train_tokens_per_second(baseline_preflight_results)
        baseline_tps_source = "paired_preflight"
    train_tps_ratio = None
    if candidate_tps is not None and baseline_tps not in (None, 0.0):
        train_tps_ratio = candidate_tps / baseline_tps

    baseline_total_tokens = derive_total_tokens(baseline_train_results)
    candidate_budget_seconds = expected_training_budget_seconds(candidate)
    projected_total_tokens = None
    if candidate_tps is not None and candidate_budget_seconds not in (None, 0.0):
        projected_total_tokens = int(candidate_tps * candidate_budget_seconds)
    projected_total_tokens_ratio = None
    if projected_total_tokens is not None and baseline_total_tokens not in (None, 0):
        projected_total_tokens_ratio = projected_total_tokens / baseline_total_tokens

    baseline_train_artifact = artifact_stats(baseline_train_results)
    baseline_preflight_artifact = artifact_stats(baseline_preflight_results)
    candidate_artifact = artifact_stats(candidate)
    artifact_margin_bytes = safe_int(preflight_policy.get("projected_artifact_margin_bytes"))
    if artifact_margin_bytes is None:
        artifact_margin_bytes = DEFAULT_PROJECTED_ARTIFACT_MARGIN_BYTES
    projection = project_artifact_bytes(
        baseline_train_artifact=baseline_train_artifact,
        baseline_preflight_artifact=baseline_preflight_artifact,
        candidate_artifact=candidate_artifact,
        margin_bytes=int(artifact_margin_bytes),
    )

    baseline_artifact = safe_int(session.get("accepted_artifact_bytes"))
    observed_artifact_gain_bytes = None
    if candidate_artifact["artifact_bytes"] is not None and baseline_artifact is not None:
        observed_artifact_gain_bytes = int(candidate_artifact["artifact_bytes"]) - int(baseline_artifact)
    projected_artifact_gain_bytes = None
    if projection.get("projected_artifact_bytes") is not None and baseline_artifact is not None:
        projected_artifact_gain_bytes = int(projection["projected_artifact_bytes"]) - int(baseline_artifact)

    hard_artifact_max = int(preflight_policy.get("hard_artifact_bytes_max", 16_000_000))
    soft_artifact_max = safe_int(preflight_policy.get("soft_artifact_bytes_max"))
    soft_artifact_min = safe_int(preflight_policy.get("soft_artifact_bytes_min"))
    min_train_ratio = safe_float(preflight_policy.get("min_train_tokens_per_second_ratio"))
    min_total_ratio = safe_float(preflight_policy.get("min_total_tokens_ratio"))
    allow_slow_if_within_soft_cap = bool(preflight_policy.get("allow_slow_if_within_soft_cap", True))
    allow_slow_if_artifact_gain_bytes = safe_int(preflight_policy.get("allow_slow_if_artifact_gain_bytes")) or 0
    require_train_tps_ratio = bool(preflight_policy.get("require_train_tokens_per_second_ratio", False))
    require_projected_total_tokens_ratio = bool(preflight_policy.get("require_projected_total_tokens_ratio", False))
    use_projected_artifact_gate = bool(preflight_policy.get("use_projected_artifact_gate", True))
    strict_projected_artifact_hard_cap = bool(preflight_policy.get("strict_projected_artifact_hard_cap", True))

    reasons: list[str] = []
    warnings: list[str] = []

    if candidate.get("status") != "success":
        reasons.append("benchmark_failed")
    if candidate.get("mode") != "benchmark":
        warnings.append(f"unexpected_mode={candidate.get('mode')}")

    observed_artifact_bytes = candidate_artifact["artifact_bytes"]
    if observed_artifact_bytes is None:
        warnings.append("missing_artifact_bytes")
    elif observed_artifact_bytes > hard_artifact_max:
        reasons.append("artifact_over_hard_cap")
    elif soft_artifact_max is not None and observed_artifact_bytes > soft_artifact_max:
        warnings.append("artifact_over_soft_cap")

    projected_artifact_bytes = safe_int(projection.get("projected_artifact_bytes"))
    if use_projected_artifact_gate and projected_artifact_bytes is None:
        warnings.append("missing_projected_artifact_bytes")
    elif use_projected_artifact_gate and projected_artifact_bytes is not None:
        if strict_projected_artifact_hard_cap and projected_artifact_bytes > hard_artifact_max:
            reasons.append("projected_artifact_over_hard_cap")
        elif soft_artifact_max is not None and projected_artifact_bytes > soft_artifact_max:
            warnings.append("projected_artifact_over_soft_cap")
        if soft_artifact_min is not None and projected_artifact_bytes < soft_artifact_min:
            warnings.append("artifact_under_soft_band")
    elif soft_artifact_min is not None and observed_artifact_bytes is not None and observed_artifact_bytes < soft_artifact_min:
        warnings.append("artifact_under_soft_band")

    if require_train_tps_ratio and train_tps_ratio is None:
        reasons.append("missing_train_tps_ratio")
    if require_projected_total_tokens_ratio and projected_total_tokens_ratio is None:
        reasons.append("missing_projected_total_tokens_ratio")

    slow_train = min_train_ratio is not None and train_tps_ratio is not None and train_tps_ratio < min_train_ratio
    slow_tokens = min_total_ratio is not None and projected_total_tokens_ratio is not None and projected_total_tokens_ratio < min_total_ratio
    enough_artifact_gain = projected_artifact_gain_bytes is not None and projected_artifact_gain_bytes >= allow_slow_if_artifact_gain_bytes
    within_soft_cap = projected_artifact_bytes is not None and (soft_artifact_max is None or projected_artifact_bytes <= soft_artifact_max)

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
            "train_tokens_per_second_source": baseline_tps_source,
            "total_tokens": baseline_total_tokens,
            "results_path": None if baseline_train_path is None else str(baseline_train_path),
            "preflight_results_path": None if baseline_preflight_path is None else str(baseline_preflight_path),
        },
        "candidate": {
            "run_id": candidate.get("run_id"),
            "val_bpb": candidate.get("val_bpb"),
            "artifact_bytes": observed_artifact_bytes,
            "projected_artifact_bytes": projected_artifact_bytes,
            "train_tokens_per_second": candidate_tps,
            "projected_total_tokens": projected_total_tokens,
            "expected_training_budget_seconds": candidate_budget_seconds,
        },
        "ratios": {
            "train_tokens_per_second": train_tps_ratio,
            "projected_total_tokens": projected_total_tokens_ratio,
        },
        "artifact_gain_bytes": observed_artifact_gain_bytes,
        "projected_artifact_gain_bytes": projected_artifact_gain_bytes,
        "artifact_projection": projection,
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
