from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train import load_and_validate_results


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_config(source_results_json: Path | None, source_config_json: Path | None) -> tuple[Path, dict[str, Any] | None]:
    if source_results_json is None and source_config_json is None:
        raise ValueError("either --source_results_json or --source_config_json is required")
    if source_results_json is not None and source_config_json is not None:
        raise ValueError("pass only one of --source_results_json or --source_config_json")
    if source_results_json is not None:
        results = dict(load_and_validate_results(source_results_json))
        config_path = Path(results["config_path"])
        return config_path, results
    assert source_config_json is not None
    return source_config_json, None


def h100_defaults(mode: str) -> dict[str, Any]:
    if mode == "8x":
        return {
            "train_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
            "val_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
            "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
            "output_dir": "./runs/runpod_h100_8x_10min/promoted",
            "results_tsv_path": "./runs/runpod_h100_8x_10min/results.tsv",
            "log_every": 20,
            "val_every": 100,
            "checkpoint_every": 100,
            "max_wallclock_seconds": 600.0,
            "use_lawa": True,
            "verify_export_reload": True,
        }
    return {
        "train_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
        "val_pattern": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
        "output_dir": "./runs/runpod_h100_1x_10min/promoted",
        "results_tsv_path": "./runs/runpod_h100_1x_10min/results.tsv",
        "log_every": 20,
        "val_every": 100,
        "checkpoint_every": 100,
        "max_wallclock_seconds": 600.0,
        "use_lawa": True,
        "verify_export_reload": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an H100-ready config from a winning 5090 run's resolved config.json.",
    )
    parser.add_argument("--source_results_json", type=str, default=None)
    parser.add_argument("--source_config_json", type=str, default=None)
    parser.add_argument("--output_config_json", type=str, required=True)
    parser.add_argument("--mode", choices=("1x", "8x"), default="8x")
    args = parser.parse_args()

    source_config_path, results = resolve_source_config(
        None if args.source_results_json is None else Path(args.source_results_json),
        None if args.source_config_json is None else Path(args.source_config_json),
    )
    cfg = load_json(source_config_path)

    for key, value in h100_defaults(args.mode).items():
        cfg[key] = value

    output_path = Path(args.output_config_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    summary = {
        "mode": args.mode,
        "source_config_json": str(source_config_path),
        "output_config_json": str(output_path),
        "effective_depth": (
            int(cfg["model"]["stem_layers"])
            + int(cfg["model"]["shared_layers"]) * int(cfg["model"]["recurrence_loops"])
            + int(cfg["model"]["tail_layers"])
        ),
        "seq_len": int(cfg["model"]["seq_len"]),
        "d_model": int(cfg["model"]["d_model"]),
        "train_batch_tokens": int(cfg["train_batch_tokens"]),
        "val_batch_tokens": int(cfg["val_batch_tokens"]),
        "grad_accum_steps": int(cfg["grad_accum_steps"]),
        "q_low_rank": int(cfg["model"].get("q_low_rank") or 0),
    }
    if results is not None:
        summary["source_run_id"] = results["run_id"]
        summary["source_val_bpb"] = results["val_bpb"]
        summary["source_artifact_bytes"] = results["artifact_bytes"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
