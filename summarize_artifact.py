from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the latest exported artifact and run results.")
    parser.add_argument("--results_json", type=str, default="./out_pgolf_recurrent_qat/results.json")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    results = load_results(results_path)
    artifact = results.get("artifact")
    if artifact is None:
        raise SystemExit("results.json does not contain an exported artifact")

    manifest = load_manifest(Path(artifact["manifest_path"]))
    print(f"compressed_model_bytes: {manifest['byte_counts']['compressed_model_bytes']}")
    print(f"code_bytes: {manifest['byte_counts']['code_bytes']}")
    print(f"artifact_bytes: {manifest['byte_counts']['artifact_bytes']}")
    print(f"param_count: {manifest['param_count']}")
    print(f"effective_depth: {manifest['effective_depth']}")
    print(f"latest_val_bpb: {results['val_bpb']}")


if __name__ == "__main__":
    main()
