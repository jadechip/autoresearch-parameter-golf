#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch}"
BASELINE_RESULTS="${BASELINE_RESULTS:-$ROOT_DIR/runs/autoresearch_5090/index/latest.json}"
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG+=(--force)
fi

echo "Initializing autoresearch session"
echo "STATE_DIR=$STATE_DIR"
echo "BASELINE_RESULTS=$BASELINE_RESULTS"

exec uv run python "$ROOT_DIR/scripts/autoresearch_state.py" \
  --state_dir "$STATE_DIR" \
  init \
  --baseline_results "$BASELINE_RESULTS" \
  "${FORCE_FLAG[@]}"
