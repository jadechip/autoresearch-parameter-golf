from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import load_and_validate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed results.json schema.")
    parser.add_argument("results_json", type=str)
    args = parser.parse_args()
    payload = load_and_validate_results(args.results_json)
    print(
        json.dumps(
            {
                "validated": True,
                "results_json": str(Path(args.results_json)),
                "run_id": payload["run_id"],
                "status": payload["status"],
                "val_bpb": payload["val_bpb"],
                "artifact_bytes": payload["artifact_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
