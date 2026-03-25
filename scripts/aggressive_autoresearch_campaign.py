from __future__ import annotations

import argparse
import json
import time
from typing import Any

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from train import atomic_write_json, load_and_validate_results


CAMPAIGN_SCHEMA_VERSION = "pgolf.aggressive_campaign.v1"

DEFAULT_RANKING_POLICY = {
    "artifact_bytes_hard_max": 16_000_000,
    "training_seconds_min_ratio": 0.70,
    "training_seconds_max_ratio": 1.40,
    "contender_bpb_gap": 0.015,
    "poor_bpb_gap": 0.025,
    "catastrophic_bpb_gap": 0.050,
    "min_attempts_before_early_stop": 2,
    "max_invalid_attempts_before_early_stop": 2,
    "max_catastrophic_attempts_before_early_stop": 2,
    "max_noncontender_attempts_before_early_stop": 3,
    "max_weak_valid_attempts_before_early_stop": 2,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_state_dir() -> Path:
    return repo_root() / ".autoresearch_aggressive"


def campaign_path(state_dir: Path) -> Path:
    return state_dir / "aggressive_campaign.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def session_path(state_dir: Path) -> Path:
    return state_dir / "session.json"


def normalize_ranking_policy(raw: Any) -> dict[str, Any]:
    payload = dict(DEFAULT_RANKING_POLICY)
    if isinstance(raw, dict):
        for key in payload:
            if key in raw:
                payload[key] = raw[key]
    return {
        "artifact_bytes_hard_max": int(payload["artifact_bytes_hard_max"]),
        "training_seconds_min_ratio": float(payload["training_seconds_min_ratio"]),
        "training_seconds_max_ratio": float(payload["training_seconds_max_ratio"]),
        "contender_bpb_gap": float(payload["contender_bpb_gap"]),
        "poor_bpb_gap": float(payload["poor_bpb_gap"]),
        "catastrophic_bpb_gap": float(payload["catastrophic_bpb_gap"]),
        "min_attempts_before_early_stop": int(payload["min_attempts_before_early_stop"]),
        "max_invalid_attempts_before_early_stop": int(payload["max_invalid_attempts_before_early_stop"]),
        "max_catastrophic_attempts_before_early_stop": int(payload["max_catastrophic_attempts_before_early_stop"]),
        "max_noncontender_attempts_before_early_stop": int(payload["max_noncontender_attempts_before_early_stop"]),
        "max_weak_valid_attempts_before_early_stop": int(payload["max_weak_valid_attempts_before_early_stop"]),
    }


def normalize_blueprint(raw: Any, index: int) -> dict[str, Any]:
    default_label = chr(ord("A") + index)
    if isinstance(raw, str):
        return {
            "label": default_label,
            "title": raw,
            "summary": raw,
            "must_change_axes": [],
        }
    if not isinstance(raw, dict):
        raise ValueError(f"unsupported aggressive blueprint payload: {raw!r}")
    return {
        "label": str(raw.get("label", default_label)),
        "title": str(raw.get("title", raw.get("summary", f"Variant {default_label}"))),
        "summary": str(raw.get("summary", raw.get("title", f"Variant {default_label}"))),
        "must_change_axes": [str(item) for item in raw.get("must_change_axes", [])],
    }


def hydrate_campaign_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    ideas_json_value = payload.get("ideas_json")
    if not ideas_json_value:
        return payload
    ideas_json_path = Path(str(ideas_json_value))
    if not ideas_json_path.is_absolute():
        ideas_json_path = repo_root() / ideas_json_path
    if not ideas_json_path.is_file():
        return payload

    source = load_json(ideas_json_path)
    changed = False
    desired_ranking_policy = normalize_ranking_policy(source.get("ranking_policy"))
    desired_default_regime = dict(source.get("campaign_default_regime", {}))
    if payload.get("ranking_policy") != desired_ranking_policy:
        payload["ranking_policy"] = desired_ranking_policy
        changed = True
    if payload.get("campaign_default_regime") != desired_default_regime:
        payload["campaign_default_regime"] = desired_default_regime
        changed = True
    source_ideas = {
        str(item["id"]): item
        for item in source.get("ideas", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    for idea in payload.get("ideas", []):
        source_idea = source_ideas.get(str(idea.get("id")))
        if source_idea is None:
            continue
        desired_title = str(source_idea.get("title", idea.get("title", "")))
        desired_branch_family = str(source_idea.get("branch_family", idea.get("branch_family", "other")))
        desired_kind = str(source_idea.get("kind", idea.get("kind", "existing_surface")))
        desired_attempt_mode = str(source_idea.get("attempt_mode", "independent_architecture_variants"))
        desired_goal = str(source_idea.get("goal", idea.get("goal", "")))
        desired_hints = [str(item) for item in source_idea.get("hints", [])]
        desired_axes = [str(item) for item in source_idea.get("must_change_axes", [])]
        desired_forbidden = [str(item) for item in source_idea.get("forbidden_refinements", [])]
        desired_attempts_allowed = int(source_idea.get("attempts_allowed", idea.get("attempts_allowed", 0) or 0))
        desired_priority = int(source_idea.get("priority", idea.get("priority", 0) or 0))
        desired_max_primary_novelty_axes = int(source_idea.get("max_primary_novelty_axes", idea.get("max_primary_novelty_axes", 1) or 1))
        desired_blueprints = [
            normalize_blueprint(item, index)
            for index, item in enumerate(source_idea.get("attempt_blueprints", []))
        ]
        if idea.get("title") != desired_title:
            idea["title"] = desired_title
            changed = True
        if idea.get("branch_family") != desired_branch_family:
            idea["branch_family"] = desired_branch_family
            changed = True
        if idea.get("kind") != desired_kind:
            idea["kind"] = desired_kind
            changed = True
        if idea.get("attempt_mode") != desired_attempt_mode:
            idea["attempt_mode"] = desired_attempt_mode
            changed = True
        if idea.get("goal") != desired_goal:
            idea["goal"] = desired_goal
            changed = True
        if idea.get("hints") != desired_hints:
            idea["hints"] = desired_hints
            changed = True
        if idea.get("must_change_axes") != desired_axes:
            idea["must_change_axes"] = desired_axes
            changed = True
        if idea.get("forbidden_refinements") != desired_forbidden:
            idea["forbidden_refinements"] = desired_forbidden
            changed = True
        if desired_attempts_allowed > 0 and idea.get("attempts_allowed") != desired_attempts_allowed:
            if int(idea.get("attempts_used", 0)) > desired_attempts_allowed:
                raise ValueError(
                    f"idea {idea.get('id')} already used {idea.get('attempts_used')} attempts but metadata now asks for {desired_attempts_allowed}"
                )
            idea["attempts_allowed"] = desired_attempts_allowed
            changed = True
        if idea.get("priority") != desired_priority:
            idea["priority"] = desired_priority
            changed = True
        if idea.get("max_primary_novelty_axes") != desired_max_primary_novelty_axes:
            idea["max_primary_novelty_axes"] = desired_max_primary_novelty_axes
            changed = True
        if idea.get("attempt_blueprints") != desired_blueprints:
            idea["attempt_blueprints"] = desired_blueprints
            changed = True
    if changed:
        payload["updated_at_unix"] = time.time()
    return payload


def write_campaign(state_dir: Path, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload["schema_version"] = CAMPAIGN_SCHEMA_VERSION
    payload["updated_at_unix"] = time.time()
    return atomic_write_json(campaign_path(state_dir), payload)


def load_campaign(state_dir: Path) -> dict[str, Any]:
    path = campaign_path(state_dir)
    if not path.is_file():
        raise ValueError(f"missing aggressive campaign file: {path}")
    payload = load_json(path)
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"unexpected aggressive campaign schema: {payload.get('schema_version')}")
    payload = hydrate_campaign_metadata(payload)
    payload = recompute_campaign_rollups(state_dir, payload)
    refresh_status(payload)
    return payload


def normalize_idea(raw: dict[str, Any], attempts_allowed: int) -> dict[str, Any]:
    resolved_attempts_allowed = int(raw.get("attempts_allowed", attempts_allowed))
    blueprints = [normalize_blueprint(item, index) for index, item in enumerate(raw.get("attempt_blueprints", []))]
    if blueprints and len(blueprints) != resolved_attempts_allowed:
        raise ValueError(
            f"idea {raw['id']} defines {len(blueprints)} attempt_blueprints but attempts_allowed={resolved_attempts_allowed}"
        )
    return {
        "id": str(raw["id"]),
        "title": str(raw["title"]),
        "branch_family": str(raw.get("branch_family", "other")),
        "kind": str(raw.get("kind", "existing_surface")),
        "attempt_mode": str(raw.get("attempt_mode", "independent_architecture_variants")),
        "goal": str(raw.get("goal", "")),
        "priority": int(raw.get("priority", 0)),
        "max_primary_novelty_axes": int(raw.get("max_primary_novelty_axes", 1)),
        "hints": [str(item) for item in raw.get("hints", [])],
        "must_change_axes": [str(item) for item in raw.get("must_change_axes", [])],
        "forbidden_refinements": [str(item) for item in raw.get("forbidden_refinements", [])],
        "attempt_blueprints": blueprints,
        "status": "pending",
        "attempts_allowed": resolved_attempts_allowed,
        "attempts_used": 0,
        "accepted_attempts": 0,
        "accepted_valid_attempts": 0,
        "accepted_invalid_attempts": 0,
        "best_val_bpb": None,
        "best_run_id": None,
        "best_observed_val_bpb": None,
        "best_observed_run_id": None,
        "best_valid_val_bpb": None,
        "best_valid_run_id": None,
        "valid_attempts": 0,
        "invalid_attempts": 0,
        "catastrophic_attempts": 0,
        "contender_attempts": 0,
        "pareto_frontier": [],
        "early_stop_reason": None,
        "attempts": [],
    }


def idea_is_done(idea: dict[str, Any]) -> bool:
    return idea.get("early_stop_reason") is not None or idea["attempts_used"] >= idea["attempts_allowed"]


def set_current_idea(campaign: dict[str, Any], index: int | None) -> None:
    campaign["current_idea_index"] = index
    for i, idea in enumerate(campaign["ideas"]):
        if index is None:
            idea["status"] = "completed" if idea_is_done(idea) else idea["status"]
            continue
        if i < index:
            idea["status"] = "completed"
        elif i == index:
            idea["status"] = "active"
        else:
            if idea_is_done(idea):
                idea["status"] = "completed"
            else:
                idea["status"] = "pending"


def current_idea(campaign: dict[str, Any]) -> dict[str, Any] | None:
    index = campaign.get("current_idea_index")
    if index is None:
        return None
    ideas = campaign.get("ideas") or []
    if not (0 <= index < len(ideas)):
        return None
    return ideas[index]


def refresh_status(campaign: dict[str, Any]) -> None:
    ideas = campaign["ideas"]
    index = campaign.get("current_idea_index")
    while index is not None and index < len(ideas) and idea_is_done(ideas[index]):
        ideas[index]["status"] = "completed"
        index += 1
    if index is None or index >= len(ideas):
        campaign["status"] = "completed"
        set_current_idea(campaign, None)
        return
    campaign["status"] = "active"
    set_current_idea(campaign, index)


def init_campaign(state_dir: Path, ideas_json: Path, tries_per_idea: int, force: bool = False) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_path(state_dir)
    if path.exists() and not force:
        raise ValueError(f"aggressive campaign already exists: {path}")

    ideas_payload = load_json(ideas_json)
    raw_ideas = ideas_payload.get("ideas")
    if not isinstance(raw_ideas, list) or not raw_ideas:
        raise ValueError(f"ideas file must contain a non-empty ideas list: {ideas_json}")

    ideas = [normalize_idea(raw, tries_per_idea) for raw in raw_ideas]
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
        "state_dir": str(state_dir),
        "ideas_json": str(ideas_json),
        "tries_per_idea_default": tries_per_idea,
        "ranking_policy": normalize_ranking_policy(ideas_payload.get("ranking_policy")),
        "campaign_default_regime": dict(ideas_payload.get("campaign_default_regime", {})),
        "status": "active",
        "current_idea_index": 0,
        "ideas": ideas,
    }
    refresh_status(payload)
    write_campaign(state_dir, payload)
    return load_campaign(state_dir)


def require_active(state_dir: Path) -> dict[str, Any]:
    payload = load_campaign(state_dir)
    if payload.get("status") != "active" or current_idea(payload) is None:
        raise ValueError("aggressive campaign is not active")
    return payload


def load_session_summary(state_dir: Path) -> dict[str, Any]:
    path = session_path(state_dir)
    if not path.is_file():
        return {}
    payload = load_json(path)
    return {
        "accepted_run_id": payload.get("accepted_run_id"),
        "accepted_val_bpb": payload.get("accepted_val_bpb"),
        "accepted_artifact_bytes": payload.get("accepted_artifact_bytes"),
        "accepted_results_path": payload.get("accepted_results_path"),
        "baseline_run_id": payload.get("baseline_run_id"),
        "baseline_val_bpb": payload.get("baseline_val_bpb"),
        "baseline_artifact_bytes": payload.get("baseline_artifact_bytes"),
        "baseline_results_path": payload.get("baseline_results_path"),
    }


def coerce_attempt_results(attempt: dict[str, Any]) -> dict[str, Any] | None:
    results_path_value = attempt.get("results_json")
    if results_path_value:
        results_path = Path(str(results_path_value))
        if not results_path.is_absolute():
            results_path = repo_root() / results_path
        if results_path.is_file():
            return dict(load_and_validate_results(results_path))
    if attempt.get("val_bpb") is None and attempt.get("artifact_bytes") is None:
        return None
    return {
        "status": "success",
        "val_bpb": attempt.get("val_bpb"),
        "artifact_bytes": attempt.get("artifact_bytes"),
        "training_seconds": attempt.get("training_seconds"),
    }


def resolve_expected_training_seconds(results: dict[str, Any]) -> float | None:
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
    value = config_payload.get("max_wallclock_seconds")
    if value is None:
        return None
    return float(value)


def assess_attempt(
    *,
    results: dict[str, Any] | None,
    ranking_policy: dict[str, Any],
    baseline_val_bpb: float | None,
) -> dict[str, Any]:
    assessment: dict[str, Any] = {
        "status": "missing_results" if results is None else str(results.get("status")),
        "artifact_valid": False,
        "training_budget_valid": False,
        "ranking_valid": False,
        "competitive_gap": None,
        "classification": "missing_results" if results is None else "unknown",
        "reasons": [],
        "expected_training_seconds": None,
        "training_seconds_ratio": None,
    }
    if results is None:
        assessment["reasons"].append("missing_results_json")
        return assessment

    artifact_bytes = results.get("artifact_bytes")
    training_seconds = results.get("training_seconds")
    val_bpb = results.get("val_bpb")
    status = str(results.get("status"))
    expected_training_seconds = resolve_expected_training_seconds(results)
    assessment["expected_training_seconds"] = expected_training_seconds

    artifact_valid = artifact_bytes is not None and int(artifact_bytes) <= int(ranking_policy["artifact_bytes_hard_max"])
    assessment["artifact_valid"] = artifact_valid
    if not artifact_valid:
        assessment["reasons"].append("artifact_over_cap")

    training_budget_valid = False
    if training_seconds is not None and expected_training_seconds is not None and expected_training_seconds > 0:
        ratio = float(training_seconds) / expected_training_seconds
        assessment["training_seconds_ratio"] = ratio
        training_budget_valid = (
            float(ranking_policy["training_seconds_min_ratio"])
            <= ratio
            <= float(ranking_policy["training_seconds_max_ratio"])
        )
    elif training_seconds is not None and expected_training_seconds is None:
        training_budget_valid = True
    assessment["training_budget_valid"] = training_budget_valid
    if not training_budget_valid:
        assessment["reasons"].append("off_budget_training_time")

    if baseline_val_bpb is not None and val_bpb is not None:
        assessment["competitive_gap"] = float(val_bpb) - float(baseline_val_bpb)

    ranking_valid = status == "success" and val_bpb is not None and artifact_valid and training_budget_valid
    assessment["ranking_valid"] = ranking_valid

    if status != "success" or val_bpb is None:
        assessment["classification"] = "failed"
    elif not artifact_valid or not training_budget_valid:
        assessment["classification"] = "invalid_budget"
    elif baseline_val_bpb is None:
        assessment["classification"] = "valid_unscoped"
    else:
        gap = float(val_bpb) - float(baseline_val_bpb)
        if gap <= float(ranking_policy["contender_bpb_gap"]):
            assessment["classification"] = "contender"
        elif gap >= float(ranking_policy["catastrophic_bpb_gap"]):
            assessment["classification"] = "catastrophic"
        elif gap >= float(ranking_policy["poor_bpb_gap"]):
            assessment["classification"] = "noncontender"
        else:
            assessment["classification"] = "valid_but_weak"
    return assessment


def dominates_attempt(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_val = left.get("val_bpb")
    right_val = right.get("val_bpb")
    left_bytes = left.get("artifact_bytes")
    right_bytes = right.get("artifact_bytes")
    if left_val is None or right_val is None or left_bytes is None or right_bytes is None:
        return False
    return (
        float(left_val) <= float(right_val)
        and int(left_bytes) <= int(right_bytes)
        and (float(left_val) < float(right_val) or int(left_bytes) < int(right_bytes))
    )


def pareto_frontier(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_attempts = [
        {
            "run_id": item.get("run_id"),
            "decision": item.get("decision"),
            "val_bpb": item.get("val_bpb"),
            "artifact_bytes": item.get("artifact_bytes"),
            "training_seconds": item.get("training_seconds"),
            "classification": item.get("classification"),
        }
        for item in attempts
        if item.get("ranking_valid") and item.get("val_bpb") is not None and item.get("artifact_bytes") is not None
    ]
    frontier: list[dict[str, Any]] = []
    for candidate in valid_attempts:
        if any(dominates_attempt(other, candidate) for other in valid_attempts if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(key=lambda item: (float(item["val_bpb"]), int(item["artifact_bytes"])))
    return frontier[:5]


def update_idea_rollups(
    idea: dict[str, Any],
    *,
    attempt: dict[str, Any],
    results: dict[str, Any] | None,
    assessment: dict[str, Any],
) -> None:
    if results is not None and results.get("val_bpb") is not None:
        best_observed = idea.get("best_observed_val_bpb")
        if best_observed is None or float(results["val_bpb"]) < float(best_observed):
            idea["best_observed_val_bpb"] = float(results["val_bpb"])
            idea["best_observed_run_id"] = attempt["run_id"]

    if assessment["ranking_valid"]:
        idea["valid_attempts"] += 1
        best_valid = idea.get("best_valid_val_bpb")
        if results is not None and results.get("val_bpb") is not None and (
            best_valid is None or float(results["val_bpb"]) < float(best_valid)
        ):
            idea["best_valid_val_bpb"] = float(results["val_bpb"])
            idea["best_valid_run_id"] = attempt["run_id"]
    else:
        idea["invalid_attempts"] += 1

    if assessment["classification"] == "catastrophic":
        idea["catastrophic_attempts"] += 1
    if assessment["classification"] == "contender":
        idea["contender_attempts"] += 1

    if attempt["decision"] == "accepted":
        idea["accepted_attempts"] += 1
        if assessment["ranking_valid"]:
            idea["accepted_valid_attempts"] += 1
        else:
            idea["accepted_invalid_attempts"] += 1
        if results is not None and results.get("val_bpb") is not None:
            best = idea.get("best_val_bpb")
            if best is None or float(results["val_bpb"]) < float(best):
                idea["best_val_bpb"] = float(results["val_bpb"])
                idea["best_run_id"] = attempt["run_id"]

    idea["pareto_frontier"] = pareto_frontier(idea["attempts"])


def reset_idea_rollups(idea: dict[str, Any]) -> None:
    idea["accepted_attempts"] = 0
    idea["accepted_valid_attempts"] = 0
    idea["accepted_invalid_attempts"] = 0
    idea["best_val_bpb"] = None
    idea["best_run_id"] = None
    idea["best_observed_val_bpb"] = None
    idea["best_observed_run_id"] = None
    idea["best_valid_val_bpb"] = None
    idea["best_valid_run_id"] = None
    idea["valid_attempts"] = 0
    idea["invalid_attempts"] = 0
    idea["catastrophic_attempts"] = 0
    idea["contender_attempts"] = 0
    idea["pareto_frontier"] = []
    idea["early_stop_reason"] = None


def recompute_campaign_rollups(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ranking_policy = normalize_ranking_policy(payload.get("ranking_policy"))
    session_summary = load_session_summary(state_dir)
    baseline_val_bpb = session_summary.get("accepted_val_bpb")
    changed = False
    for idea in payload.get("ideas", []):
        original_snapshot = json.dumps(
            {
                key: idea.get(key)
                for key in (
                    "accepted_attempts",
                    "accepted_valid_attempts",
                    "accepted_invalid_attempts",
                    "best_val_bpb",
                    "best_run_id",
                    "best_observed_val_bpb",
                    "best_observed_run_id",
                    "best_valid_val_bpb",
                    "best_valid_run_id",
                    "valid_attempts",
                    "invalid_attempts",
                    "catastrophic_attempts",
                    "contender_attempts",
                    "pareto_frontier",
                    "early_stop_reason",
                )
            },
            sort_keys=True,
        )
        reset_idea_rollups(idea)
        normalized_attempts: list[dict[str, Any]] = []
        for attempt in idea.get("attempts", []):
            attempt_payload = dict(attempt)
            results = coerce_attempt_results(attempt_payload)
            assessment = assess_attempt(results=results, ranking_policy=ranking_policy, baseline_val_bpb=baseline_val_bpb)
            attempt_payload["classification"] = assessment["classification"]
            attempt_payload["ranking_valid"] = assessment["ranking_valid"]
            attempt_payload["artifact_valid"] = assessment["artifact_valid"]
            attempt_payload["training_budget_valid"] = assessment["training_budget_valid"]
            attempt_payload["competitive_gap"] = assessment["competitive_gap"]
            attempt_payload["expected_training_seconds"] = assessment["expected_training_seconds"]
            attempt_payload["training_seconds_ratio"] = assessment["training_seconds_ratio"]
            attempt_payload["policy_violation"] = attempt_payload.get("decision") == "accepted" and not assessment["ranking_valid"]
            attempt_payload["reasons"] = list(assessment["reasons"])
            normalized_attempts.append(attempt_payload)
            update_idea_rollups(idea, attempt=attempt_payload, results=results, assessment=assessment)
        idea["attempts"] = normalized_attempts
        maybe_mark_early_stop(payload, idea, baseline_val_bpb=baseline_val_bpb)
        refreshed_snapshot = json.dumps(
            {
                key: idea.get(key)
                for key in (
                    "accepted_attempts",
                    "accepted_valid_attempts",
                    "accepted_invalid_attempts",
                    "best_val_bpb",
                    "best_run_id",
                    "best_observed_val_bpb",
                    "best_observed_run_id",
                    "best_valid_val_bpb",
                    "best_valid_run_id",
                    "valid_attempts",
                    "invalid_attempts",
                    "catastrophic_attempts",
                    "contender_attempts",
                    "pareto_frontier",
                    "early_stop_reason",
                )
            },
            sort_keys=True,
        )
        if refreshed_snapshot != original_snapshot:
            changed = True
    if changed:
        payload["updated_at_unix"] = time.time()
    return payload


def weak_valid_attempts(idea: dict[str, Any]) -> int:
    return sum(1 for attempt in idea.get("attempts", []) if attempt.get("classification") in {"valid_but_weak", "noncontender"})


def has_invalid_near_miss(idea: dict[str, Any], policy: dict[str, Any]) -> bool:
    contender_gap = float(policy["contender_bpb_gap"])
    for attempt in idea.get("attempts", []):
        if attempt.get("classification") != "invalid_budget":
            continue
        gap = attempt.get("competitive_gap")
        if gap is not None and float(gap) <= contender_gap:
            return True
    return False


def maybe_mark_early_stop(
    campaign: dict[str, Any],
    idea: dict[str, Any],
    *,
    baseline_val_bpb: float | None,
) -> None:
    if idea.get("early_stop_reason") is not None:
        return
    attempts_used = int(idea.get("attempts_used", 0))
    policy = campaign.get("ranking_policy") or DEFAULT_RANKING_POLICY
    if attempts_used < int(policy["min_attempts_before_early_stop"]):
        return

    if idea.get("contender_attempts", 0) > 0:
        return

    invalid_near_miss = has_invalid_near_miss(idea, policy)
    weak_valid_count = weak_valid_attempts(idea)

    if not invalid_near_miss and idea.get("invalid_attempts", 0) >= int(policy["max_invalid_attempts_before_early_stop"]):
        idea["early_stop_reason"] = "too_many_invalid_attempts"
        return

    if idea.get("catastrophic_attempts", 0) >= int(policy["max_catastrophic_attempts_before_early_stop"]):
        idea["early_stop_reason"] = "too_many_catastrophic_attempts"
        return

    if weak_valid_count >= int(policy["max_weak_valid_attempts_before_early_stop"]):
        best_valid = idea.get("best_valid_val_bpb")
        if best_valid is None or baseline_val_bpb is None:
            idea["early_stop_reason"] = "repeated_weak_valid_attempts"
            return
        if float(best_valid) - float(baseline_val_bpb) >= float(policy["poor_bpb_gap"]):
            idea["early_stop_reason"] = "repeated_weak_valid_attempts"
            return

    if attempts_used >= int(policy["max_noncontender_attempts_before_early_stop"]):
        best_valid = idea.get("best_valid_val_bpb")
        if best_valid is None:
            if invalid_near_miss:
                return
            idea["early_stop_reason"] = "no_valid_contender_after_multiple_attempts"
            return
        if baseline_val_bpb is not None and (
            float(best_valid) - float(baseline_val_bpb) >= float(policy["poor_bpb_gap"])
        ):
            if invalid_near_miss:
                return
            idea["early_stop_reason"] = "best_valid_run_still_far_off_frontier"


def record_attempt(
    state_dir: Path,
    *,
    run_id: str,
    decision: str,
    results_json: Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if decision not in {"accepted", "reverted"}:
        raise ValueError(f"unsupported decision: {decision}")

    payload = require_active(state_dir)
    idea = current_idea(payload)
    if idea is None:
        raise ValueError("no active idea to record")

    ranking_policy = normalize_ranking_policy(payload.get("ranking_policy"))
    session_summary = load_session_summary(state_dir)
    baseline_val_bpb = session_summary.get("accepted_val_bpb")
    results: dict[str, Any] | None = None
    if results_json is not None:
        results = dict(load_and_validate_results(results_json))
    assessment = assess_attempt(results=results, ranking_policy=ranking_policy, baseline_val_bpb=baseline_val_bpb)

    attempt = {
        "recorded_at_unix": time.time(),
        "run_id": run_id,
        "decision": decision,
        "results_json": None if results_json is None else str(results_json),
        "val_bpb": None if results is None else results.get("val_bpb"),
        "artifact_bytes": None if results is None else results.get("artifact_bytes"),
        "training_seconds": None if results is None else results.get("training_seconds"),
        "classification": assessment["classification"],
        "ranking_valid": assessment["ranking_valid"],
        "artifact_valid": assessment["artifact_valid"],
        "training_budget_valid": assessment["training_budget_valid"],
        "competitive_gap": assessment["competitive_gap"],
        "expected_training_seconds": assessment["expected_training_seconds"],
        "training_seconds_ratio": assessment["training_seconds_ratio"],
        "policy_violation": decision == "accepted" and not assessment["ranking_valid"],
        "reasons": list(assessment["reasons"]),
        "notes": notes,
    }
    idea["attempts"].append(attempt)
    idea["attempts_used"] += 1
    update_idea_rollups(idea, attempt=attempt, results=results, assessment=assessment)
    maybe_mark_early_stop(payload, idea, baseline_val_bpb=baseline_val_bpb)

    refresh_status(payload)
    write_campaign(state_dir, payload)
    return load_campaign(state_dir)


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def idea_runtime_view(campaign: dict[str, Any], idea: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    payload = dict(idea)
    blueprints = [dict(item) for item in idea.get("attempt_blueprints", [])]
    attempts_used = int(idea.get("attempts_used", 0))
    attempts_allowed = int(idea.get("attempts_allowed", 0))
    next_index = attempts_used if attempts_used < len(blueprints) else None
    session_summary = load_session_summary(state_dir)
    best_valid = idea.get("best_valid_val_bpb")
    baseline_val_bpb = session_summary.get("accepted_val_bpb")
    ranking_policy = dict(campaign.get("ranking_policy", DEFAULT_RANKING_POLICY))
    contender_found = idea.get("contender_attempts", 0) > 0
    invalid_near_miss = has_invalid_near_miss(idea, ranking_policy)
    payload["attempts_remaining"] = max(0, attempts_allowed - attempts_used)
    payload["next_attempt_number"] = None if next_index is None else attempts_used + 1
    payload["completed_blueprints"] = blueprints[:attempts_used]
    payload["remaining_blueprints"] = blueprints[attempts_used:]
    payload["next_attempt_blueprint"] = None if next_index is None else blueprints[next_index]
    payload["ranking_policy"] = ranking_policy
    payload["campaign_default_regime"] = dict(campaign.get("campaign_default_regime", {}))
    payload["accepted_baseline"] = session_summary
    payload["frontier_gap_to_baseline"] = None if best_valid is None or baseline_val_bpb is None else float(best_valid) - float(baseline_val_bpb)
    payload["weak_valid_attempts"] = weak_valid_attempts(idea)
    payload["invalid_near_miss_detected"] = invalid_near_miss
    payload["recent_attempts"] = [
        {
            "run_id": item.get("run_id"),
            "classification": item.get("classification"),
            "val_bpb": item.get("val_bpb"),
            "artifact_bytes": item.get("artifact_bytes"),
            "competitive_gap": item.get("competitive_gap"),
        }
        for item in idea.get("attempts", [])[-3:]
    ]
    if contender_found:
        payload["recommended_phase"] = "refine"
    elif invalid_near_miss:
        payload["recommended_phase"] = "repair"
    else:
        payload["recommended_phase"] = "establish"
    payload["recommended_experiment_style"] = {
        "anchor_config_required": True,
        "primary_novelty_axes": int(idea.get("max_primary_novelty_axes", 1) or 1),
        "prefer_single_support_change": True,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="State for the aggressive idea-by-idea autoresearch campaign.")
    parser.add_argument("--state_dir", type=str, default=str(default_state_dir()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--ideas_json", type=str, required=True)
    init_parser.add_argument("--tries_per_idea", type=int, default=6)
    init_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("show")
    subparsers.add_parser("current")
    subparsers.add_parser("require-active")

    record_parser = subparsers.add_parser("record-attempt")
    record_parser.add_argument("--run_id", type=str, required=True)
    record_parser.add_argument("--decision", type=str, choices=("accepted", "reverted"), required=True)
    record_parser.add_argument("--results_json", type=str, default=None)
    record_parser.add_argument("--notes", type=str, default=None)

    args = parser.parse_args()
    state_dir = Path(args.state_dir)

    if args.command == "init":
        print_payload(init_campaign(state_dir, Path(args.ideas_json), args.tries_per_idea, force=args.force))
        return
    if args.command == "show":
        print_payload(load_campaign(state_dir))
        return
    if args.command == "current":
        payload = require_active(state_dir)
        idea = current_idea(payload)
        if idea is None:
            raise ValueError("no active idea")
        print_payload(idea_runtime_view(payload, idea, state_dir))
        return
    if args.command == "require-active":
        print_payload(require_active(state_dir))
        return
    if args.command == "record-attempt":
        print_payload(
            record_attempt(
                state_dir,
                run_id=args.run_id,
                decision=args.decision,
                results_json=None if args.results_json is None else Path(args.results_json),
                notes=args.notes,
            )
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
