from __future__ import annotations

import argparse
import json
import os
from typing import Any

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from train import atomic_write_json, load_and_validate_results


DEFAULT_ARTIFACT_HARD_MAX = 16_000_000
DEFAULT_TRAINING_SECONDS_MIN_RATIO = 0.70
DEFAULT_TRAINING_SECONDS_MAX_RATIO = 1.40


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


def resolve_expected_training_seconds(results: dict) -> float | None:
    config_path_value = results.get("config_path")
    if not config_path_value:
        return None
    config_path = Path(str(config_path_value))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.is_file():
        return None
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return safe_float(config_payload.get("max_wallclock_seconds"))


def basic_index_assessment(results: dict) -> dict[str, Any]:
    artifact_bytes = safe_int(results.get("artifact_bytes"))
    expected_training_seconds = resolve_expected_training_seconds(results)
    training_seconds = safe_float(results.get("training_seconds"))
    artifact_valid = artifact_bytes is not None and artifact_bytes <= DEFAULT_ARTIFACT_HARD_MAX
    training_budget_valid = True
    training_seconds_ratio = None
    if training_seconds is not None and expected_training_seconds not in (None, 0.0):
        training_seconds_ratio = training_seconds / expected_training_seconds
        training_budget_valid = (
            DEFAULT_TRAINING_SECONDS_MIN_RATIO <= training_seconds_ratio <= DEFAULT_TRAINING_SECONDS_MAX_RATIO
        )
    ranking_valid = (
        results.get("status") == "success"
        and results.get("mode") == "train"
        and results.get("val_bpb") is not None
        and artifact_valid
        and training_budget_valid
    )
    return {
        "artifact_valid": artifact_valid,
        "training_budget_valid": training_budget_valid,
        "ranking_valid": ranking_valid,
        "training_seconds_ratio": training_seconds_ratio,
        "expected_training_seconds": expected_training_seconds,
        "artifact_bytes_hard_max": DEFAULT_ARTIFACT_HARD_MAX,
    }


def with_source(payload: dict, results_path: Path) -> dict:
    enriched = dict(payload)
    enriched["indexed_source_results_path"] = str(results_path)
    enriched["index_assessment"] = basic_index_assessment(enriched)
    return enriched


def maybe_symlink(link_path: Path, target_path: Path) -> None:
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(target_path, link_path, target_is_directory=True)
    except OSError:
        pass


def remove_if_exists(path: Path) -> None:
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
    except OSError:
        pass


def load_existing_best(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_best(path: Path, run_link_path: Path) -> dict | None:
    best = load_existing_best(path)
    if best is None:
        return None
    if basic_index_assessment(best).get("ranking_valid"):
        return best
    remove_if_exists(path)
    remove_if_exists(run_link_path)
    return None


def is_better_valid(current: dict, best: dict | None) -> bool:
    if not basic_index_assessment(current).get("ranking_valid"):
        return False
    if best is None:
        return True
    best_bpb = best.get("val_bpb")
    if best_bpb is None:
        return True
    return float(current["val_bpb"]) < float(best_bpb)


def is_better_raw(current: dict, best: dict | None) -> bool:
    if current.get("status") != "success" or current.get("val_bpb") is None:
        return False
    if current.get("mode") != "train":
        return False
    if best is None:
        return True
    best_bpb = best.get("val_bpb")
    if best_bpb is None:
        return True
    return float(current["val_bpb"]) < float(best_bpb)


def index_run(results_json: Path, index_dir: Path) -> dict[str, str]:
    results = dict(load_and_validate_results(results_json))
    index_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(results["output_dir"])

    latest_path = index_dir / "latest.json"
    best_path = index_dir / "best.json"
    best_valid_path = index_dir / "best_valid.json"
    best_raw_path = index_dir / "best_raw.json"
    atomic_write_json(latest_path, with_source(results, results_json))
    maybe_symlink(index_dir / "latest_run", run_dir)

    existing_best = sanitize_best(best_path, index_dir / "best_run")
    existing_best_valid = sanitize_best(best_valid_path, index_dir / "best_valid_run")
    valid_updated = False
    if is_better_valid(results, existing_best):
        atomic_write_json(best_path, with_source(results, results_json))
        maybe_symlink(index_dir / "best_run", run_dir)
        valid_updated = True
    if is_better_valid(results, existing_best_valid):
        atomic_write_json(best_valid_path, with_source(results, results_json))
        maybe_symlink(index_dir / "best_valid_run", run_dir)

    existing_best_raw = load_existing_best(best_raw_path)
    raw_updated = False
    if is_better_raw(results, existing_best_raw):
        atomic_write_json(best_raw_path, with_source(results, results_json))
        maybe_symlink(index_dir / "best_raw_run", run_dir)
        raw_updated = True

    return {
        "latest_json": str(latest_path),
        "best_json": str(best_path),
        "best_valid_json": str(best_valid_path),
        "best_raw_json": str(best_raw_path),
        "run_dir": str(run_dir),
        "latest_mode": str(results.get("mode")),
        "best_updated": str(valid_updated).lower(),
        "best_raw_updated": str(raw_updated).lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update latest/best autoresearch indices from a completed run.")
    parser.add_argument("results_json", type=str)
    parser.add_argument("--index_dir", type=str, required=True)
    args = parser.parse_args()
    payload = index_run(Path(args.results_json), Path(args.index_dir))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
