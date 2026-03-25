#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.minimal_autoresearch}"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$ROOT_DIR/runs/minimal_autoresearch_5090}"
RUNS_DIR="${RUNS_DIR:-$AUTORESEARCH_ROOT/runs}"
INDEX_DIR="${INDEX_DIR:-$AUTORESEARCH_ROOT/index}"
LAST_RUN_JSON="${LAST_RUN_JSON:-$AUTORESEARCH_ROOT/last_run.json}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$AUTORESEARCH_ROOT/results.tsv}"
DEFAULT_CONFIG_JSON="$ROOT_DIR/configs/autoresearch_5090_frontier_5min.json"
CONFIG_JSON="${CONFIG_JSON:-}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
RUN_ID="${RUN_ID:-mini-ar-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/$RUN_ID}"
LOCK_DIR="$AUTORESEARCH_ROOT/.lock"
status=0
indexed="false"
RESULTS_JSON=""

mkdir -p "$RUNS_DIR" "$INDEX_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

resolve_config_json() {
  if [[ -n "$CONFIG_JSON" ]]; then
    return 0
  fi
  if [[ -f "$STATE_DIR/state.json" ]]; then
    CONFIG_JSON="$($PYTHON_BIN - <<'PY' "$STATE_DIR/state.json"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload.get("config_json") or "")
PY
)"
  fi
  if [[ -z "$CONFIG_JSON" ]]; then
    CONFIG_JSON="$DEFAULT_CONFIG_JSON"
  fi
}

resolve_config_json

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "Missing config json: $CONFIG_JSON" >&2
  exit 2
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another minimal autoresearch experiment appears to be running: $LOCK_DIR" >&2
  exit 2
fi

write_last_run() {
  "$PYTHON_BIN" - <<'PY' "$LAST_RUN_JSON" "$RUN_ID" "$OUTPUT_DIR" "$RESULTS_JSON" "$status" "$indexed" "$CONFIG_JSON"
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "output_dir": sys.argv[3],
    "results_json": sys.argv[4] if sys.argv[4] and Path(sys.argv[4]).is_file() else None,
    "exit_code": int(sys.argv[5]),
    "indexed": sys.argv[6] == "true",
    "config_json": sys.argv[7],
    "finished_at_unix": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

cleanup() {
  status=$?
  if [[ -x "$PYTHON_BIN" ]]; then
    write_last_run || true
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

echo "Starting minimal autoresearch experiment"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"

set +e
CONFIG_JSON="$CONFIG_JSON" \
OUTPUT_DIR="$OUTPUT_DIR" \
RESULTS_TSV_PATH="$RESULTS_TSV_PATH" \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
bash "$ROOT_DIR/scripts/runpod_5090_train.sh" --run_name "$RUN_ID" "$@"
status=$?
set -e

RESULTS_JSON="$OUTPUT_DIR/results.json"
if [[ -f "$RESULTS_JSON" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/index_autoresearch_run.py" "$RESULTS_JSON" --index_dir "$INDEX_DIR" >/dev/null
  indexed="true"
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "Minimal experiment finished"
echo "Last run: $LAST_RUN_JSON"
echo "Results: $RESULTS_JSON"
