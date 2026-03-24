from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from train import atomic_write_json, load_and_validate_results


def with_source(payload: dict, results_path: Path) -> dict:
    enriched = dict(payload)
    enriched["indexed_source_results_path"] = str(results_path)
    return enriched


def maybe_symlink(link_path: Path, target_path: Path) -> None:
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(target_path, link_path, target_is_directory=True)
    except OSError:
        pass


def load_existing_best(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_better(current: dict, best: dict | None) -> bool:
    if current.get("status") != "success" or current.get("val_bpb") is None:
        return False
    if current.get("mode") != "train":
        return False
    preflight = current.get("preflight")
    if isinstance(preflight, dict) and preflight.get("decision") == "skip":
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
    atomic_write_json(latest_path, with_source(results, results_json))
    maybe_symlink(index_dir / "latest_run", run_dir)

    existing_best = load_existing_best(best_path)
    if is_better(results, existing_best):
        atomic_write_json(best_path, with_source(results, results_json))
        maybe_symlink(index_dir / "best_run", run_dir)

    return {
        "latest_json": str(latest_path),
        "best_json": str(best_path),
        "run_dir": str(run_dir),
        "latest_mode": str(results.get("mode")),
        "best_updated": str(is_better(results, existing_best)).lower(),
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
