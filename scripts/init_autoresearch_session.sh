#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch}"
BASELINE_RESULTS="${BASELINE_RESULTS:-$ROOT_DIR/runs/autoresearch_5090/index/latest.json}"
TRACKED_ACCEPTED_STATE="${TRACKED_ACCEPTED_STATE:-$ROOT_DIR/state/autoresearch/accepted_state.json}"
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG+=(--force)
fi

echo "Initializing autoresearch session"
echo "STATE_DIR=$STATE_DIR"
echo "BASELINE_RESULTS=$BASELINE_RESULTS"
echo "TRACKED_ACCEPTED_STATE=$TRACKED_ACCEPTED_STATE"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

if [[ -f "$BASELINE_RESULTS" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" \
    --state_dir "$STATE_DIR" \
    init \
    --baseline_results "$BASELINE_RESULTS" \
    "${FORCE_FLAG[@]}"
elif [[ -f "$TRACKED_ACCEPTED_STATE" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" \
    --state_dir "$STATE_DIR" \
    init-from-tracked \
    --tracked_state "$TRACKED_ACCEPTED_STATE" \
    "${FORCE_FLAG[@]}"
else
  echo "Missing both baseline results and tracked accepted state." >&2
  echo "Expected either: $BASELINE_RESULTS" >&2
  echo "or: $TRACKED_ACCEPTED_STATE" >&2
  exit 2
fi
