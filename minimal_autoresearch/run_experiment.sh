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
REQUESTED_CONFIG_JSON="${CONFIG_JSON:-}"
REQUESTED_MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
CONFIG_JSON=""
MAX_WALLCLOCK_SECONDS=""
RUN_ID="${RUN_ID:-mini-ar-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/$RUN_ID}"
LOCK_DIR="$AUTORESEARCH_ROOT/.lock"
LAUNCH_CONFIG_JSON=""
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
  if [[ -f "$STATE_DIR/state.json" ]]; then
    CONFIG_JSON="$($PYTHON_BIN - <<'PY' "$STATE_DIR/state.json"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload.get("config_json") or "")
PY
)"
  fi
  if [[ -z "$CONFIG_JSON" && -n "$REQUESTED_CONFIG_JSON" ]]; then
    CONFIG_JSON="$REQUESTED_CONFIG_JSON"
  fi
  if [[ -z "$CONFIG_JSON" ]]; then
    CONFIG_JSON="$DEFAULT_CONFIG_JSON"
  fi
}

resolve_wallclock_seconds() {
  MAX_WALLCLOCK_SECONDS="$REQUESTED_MAX_WALLCLOCK_SECONDS"
}

resolve_config_json
resolve_wallclock_seconds

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "Missing config json: $CONFIG_JSON" >&2
  exit 2
fi

if [[ $# -ne 0 ]]; then
  echo "minimal_autoresearch/run_experiment.sh does not accept extra train arguments" >&2
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

mkdir -p "$OUTPUT_DIR"

prepare_launch_config() {
  if [[ ! -f "$STATE_DIR/state.json" ]]; then
    return 0
  fi
  LAUNCH_CONFIG_JSON="$OUTPUT_DIR/launch_config.json"
  mapfile -t launch_meta < <("$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$STATE_DIR" "$CONFIG_JSON" "$LAUNCH_CONFIG_JSON"
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
state_dir = Path(sys.argv[2])
config_json = sys.argv[3]
output_path = Path(sys.argv[4])

sys.path.insert(0, str(root_dir))

from minimal_autoresearch.state import atomic_write_json, load_json, load_state, protocol_from_state, resolve_repo_path

state = load_state(state_dir)
candidate_config = load_json(resolve_repo_path(config_json))
protocol = protocol_from_state(state)
for field, value in protocol.items():
    candidate_config[field] = value
atomic_write_json(output_path, candidate_config)
print(str(output_path))
print(str(protocol["max_wallclock_seconds"]))
PY
)
  LAUNCH_CONFIG_JSON="${launch_meta[0]}"
  MAX_WALLCLOCK_SECONDS="${launch_meta[1]}"
  CONFIG_JSON="$LAUNCH_CONFIG_JSON"
}

prepare_launch_config

write_last_run() {
  "$PYTHON_BIN" - <<'PY' "$LAST_RUN_JSON" "$RUN_ID" "$OUTPUT_DIR" "$RESULTS_JSON" "$status" "$indexed" "$CONFIG_JSON" "$LAUNCH_CONFIG_JSON" "$MAX_WALLCLOCK_SECONDS"
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
    "launch_config_json": sys.argv[8] or None,
    "max_wallclock_seconds": float(sys.argv[9]) if sys.argv[9] else None,
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
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
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
