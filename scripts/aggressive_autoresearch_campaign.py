from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from train import atomic_write_json, load_and_validate_results


CAMPAIGN_SCHEMA_VERSION = "pgolf.aggressive_campaign.v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_state_dir() -> Path:
    return repo_root() / ".autoresearch_aggressive"


def campaign_path(state_dir: Path) -> Path:
    return state_dir / "aggressive_campaign.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    source_ideas = {
        str(item["id"]): item
        for item in source.get("ideas", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    changed = False
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
        "hints": [str(item) for item in raw.get("hints", [])],
        "must_change_axes": [str(item) for item in raw.get("must_change_axes", [])],
        "forbidden_refinements": [str(item) for item in raw.get("forbidden_refinements", [])],
        "attempt_blueprints": blueprints,
        "status": "pending",
        "attempts_allowed": resolved_attempts_allowed,
        "attempts_used": 0,
        "accepted_attempts": 0,
        "best_val_bpb": None,
        "best_run_id": None,
        "attempts": [],
    }


def set_current_idea(campaign: dict[str, Any], index: int | None) -> None:
    campaign["current_idea_index"] = index
    for i, idea in enumerate(campaign["ideas"]):
        if index is None:
            idea["status"] = "completed" if idea["attempts_used"] >= idea["attempts_allowed"] else idea["status"]
            continue
        if i < index:
            idea["status"] = "completed"
        elif i == index:
            idea["status"] = "active"
        else:
            if idea["attempts_used"] >= idea["attempts_allowed"]:
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
    while index is not None and index < len(ideas) and ideas[index]["attempts_used"] >= ideas[index]["attempts_allowed"]:
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

    results: dict[str, Any] | None = None
    if results_json is not None:
        results = dict(load_and_validate_results(results_json))

    attempt = {
        "recorded_at_unix": time.time(),
        "run_id": run_id,
        "decision": decision,
        "results_json": None if results_json is None else str(results_json),
        "val_bpb": None if results is None else results.get("val_bpb"),
        "artifact_bytes": None if results is None else results.get("artifact_bytes"),
        "training_seconds": None if results is None else results.get("training_seconds"),
        "notes": notes,
    }
    idea["attempts"].append(attempt)
    idea["attempts_used"] += 1
    if decision == "accepted":
        idea["accepted_attempts"] += 1
        if results is not None:
            best = idea.get("best_val_bpb")
            if best is None or results["val_bpb"] < best:
                idea["best_val_bpb"] = results["val_bpb"]
                idea["best_run_id"] = run_id

    refresh_status(payload)
    write_campaign(state_dir, payload)
    return load_campaign(state_dir)


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def idea_runtime_view(idea: dict[str, Any]) -> dict[str, Any]:
    payload = dict(idea)
    blueprints = [dict(item) for item in idea.get("attempt_blueprints", [])]
    attempts_used = int(idea.get("attempts_used", 0))
    attempts_allowed = int(idea.get("attempts_allowed", 0))
    next_index = attempts_used if attempts_used < len(blueprints) else None
    payload["attempts_remaining"] = max(0, attempts_allowed - attempts_used)
    payload["next_attempt_number"] = None if next_index is None else attempts_used + 1
    payload["completed_blueprints"] = blueprints[:attempts_used]
    payload["remaining_blueprints"] = blueprints[attempts_used:]
    payload["next_attempt_blueprint"] = None if next_index is None else blueprints[next_index]
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
        print_payload(idea_runtime_view(idea))
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
