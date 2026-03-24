#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$ROOT_DIR/runs/autoresearch_5090}"
RUNS_DIR="${RUNS_DIR:-$AUTORESEARCH_ROOT/runs}"
INDEX_DIR="${INDEX_DIR:-$AUTORESEARCH_ROOT/index}"
CONFIG_JSON="${CONFIG_JSON:-$ROOT_DIR/configs/autoresearch_5090_5min.json}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$AUTORESEARCH_ROOT/results.tsv}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
RUN_ID="${RUN_ID:-ar5090-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/$RUN_ID}"
PRE_OUTPUT_DIR="${PRE_OUTPUT_DIR:-$RUNS_DIR/${RUN_ID}-preflight}"
LOCK_DIR="$AUTORESEARCH_ROOT/.lock"
ACTIVE_JSON="$INDEX_DIR/active.json"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch}"
SESSION_JSON="$STATE_DIR/session.json"
ALLOW_UNINITIALIZED_SESSION="${ALLOW_UNINITIALIZED_SESSION:-0}"
PREFLIGHT_DECISION_JSON="$PRE_OUTPUT_DIR/preflight_decision.json"
STARTED_SESSION_RUN=0

mkdir -p "$RUNS_DIR" "$INDEX_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ -e "$PRE_OUTPUT_DIR" ]]; then
  echo "Preflight output directory already exists: $PRE_OUTPUT_DIR" >&2
  exit 2
fi

USE_SESSION_STATE=0
if [[ -f "$SESSION_JSON" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" require-ready >/dev/null
  USE_SESSION_STATE=1
elif [[ "$ALLOW_UNINITIALIZED_SESSION" == "1" || "$RUN_ID" == baseline_* ]]; then
  echo "No autoresearch session found at $SESSION_JSON; allowing baseline/manual run." >&2
  echo "After the baseline, run: bash scripts/init_autoresearch_session.sh" >&2
else
  echo "No autoresearch session found at $SESSION_JSON" >&2
  echo "Run a baseline first (for example RUN_ID=baseline_5090_5min bash scripts/run_autoresearch_experiment.sh)," >&2
  echo "then initialize session state with: bash scripts/init_autoresearch_session.sh" >&2
  exit 2
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another autoresearch experiment appears to be running: $LOCK_DIR" >&2
  exit 2
fi

cleanup() {
  if [[ "$USE_SESSION_STATE" == "1" && "$STARTED_SESSION_RUN" == "1" && ! -f "$OUTPUT_DIR/results.json" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" abort-run --run_id "$RUN_ID" --reason "wrapper_cleanup" >/dev/null 2>&1 || true
  fi
  rm -f "$ACTIVE_JSON"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

run_preflight_if_enabled() {
  if [[ "$USE_SESSION_STATE" != "1" ]]; then
    return 0
  fi

  local preflight_enabled
  preflight_enabled="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
payload = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print('1' if ((payload.get('search_policy') or {}).get('preflight') or {}).get('enabled') else '0')
PY
)"
  if [[ "$preflight_enabled" != "1" ]]; then
    return 0
  fi

  local benchmark_train_steps no_lawa no_eval_first disable_reload_verify preflight_exit_code
  benchmark_train_steps="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
policy = (json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('search_policy') or {}).get('preflight') or {}
print(int(policy.get('benchmark_train_steps', 2)))
PY
)"
  no_lawa="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
policy = (json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('search_policy') or {}).get('preflight') or {}
print('1' if policy.get('no_lawa', True) else '0')
PY
)"
  no_eval_first="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
policy = (json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('search_policy') or {}).get('preflight') or {}
print('1' if policy.get('no_eval_first_step', True) else '0')
PY
)"
  disable_reload_verify="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
policy = (json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('search_policy') or {}).get('preflight') or {}
print('1' if policy.get('disable_reload_verify', True) else '0')
PY
)"
  preflight_exit_code="$($PYTHON_BIN - <<'PY' "$SESSION_JSON"
import json, sys
policy = (json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('search_policy') or {}).get('preflight') or {}
print(int(policy.get('preflight_exit_code', 10)))
PY
)"

  echo "Running benchmark preflight"
  echo "PRE_OUTPUT_DIR=$PRE_OUTPUT_DIR"
  local benchmark_args=(--benchmark --benchmark_train_steps "$benchmark_train_steps" --val_every 999999 --run_name "${RUN_ID}-preflight")
  if [[ "$no_lawa" == "1" ]]; then
    benchmark_args+=(--no_lawa)
  fi
  if [[ "$no_eval_first" == "1" ]]; then
    benchmark_args+=(--no_eval_first_step)
  fi
  if [[ "$disable_reload_verify" == "1" ]]; then
    benchmark_args+=(--no_verify_export_reload)
  fi

  CONFIG_JSON="$CONFIG_JSON" \
  OUTPUT_DIR="$PRE_OUTPUT_DIR" \
  RESULTS_TSV_PATH="$RESULTS_TSV_PATH" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  bash "$ROOT_DIR/scripts/runpod_5090_train.sh" "${benchmark_args[@]}" "$@"

  local preflight_results="$PRE_OUTPUT_DIR/results.json"
  if [[ ! -f "$preflight_results" ]]; then
    echo "Missing benchmark preflight results: $preflight_results" >&2
    return 1
  fi

  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_preflight.py" \
    --state_dir "$STATE_DIR" \
    --results_json "$preflight_results" \
    --write_json "$PREFLIGHT_DECISION_JSON" \
    --attach_to_results >/dev/null

  "$PYTHON_BIN" "$ROOT_DIR/scripts/index_autoresearch_run.py" "$preflight_results" --index_dir "$INDEX_DIR" >/dev/null

  local preflight_decision
  preflight_decision="$($PYTHON_BIN - <<'PY' "$PREFLIGHT_DECISION_JSON"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('decision', 'proceed'))
PY
)"

  if [[ "$preflight_decision" == "skip" ]]; then
    echo "Preflight rejected full proxy run"
    echo "Preflight results: $preflight_results"
    echo "Preflight decision: $PREFLIGHT_DECISION_JSON"
    exit "$preflight_exit_code"
  fi
}

echo "Starting autoresearch experiment"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "INDEX_DIR=$INDEX_DIR"
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"

run_preflight_if_enabled "$@"

cat > "$ACTIVE_JSON" <<EOF2
{
  "status": "running",
  "run_id": "$RUN_ID",
  "output_dir": "$OUTPUT_DIR",
  "results_path": "$OUTPUT_DIR/results.json",
  "metrics_path": "$OUTPUT_DIR/metrics.jsonl",
  "tensorboard_log_dir": "$OUTPUT_DIR/tensorboard",
  "crash_path": "$OUTPUT_DIR/crash.json",
  "preflight_output_dir": "$PRE_OUTPUT_DIR",
  "preflight_decision_path": "$PREFLIGHT_DECISION_JSON"
}
EOF2

if [[ "$USE_SESSION_STATE" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" \
    --state_dir "$STATE_DIR" \
    start-run \
    --run_id "$RUN_ID" \
    --output_dir "$OUTPUT_DIR" \
    --results_path "$OUTPUT_DIR/results.json" \
    --metrics_path "$OUTPUT_DIR/metrics.jsonl" \
    --tensorboard_log_dir "$OUTPUT_DIR/tensorboard" \
    --crash_path "$OUTPUT_DIR/crash.json" >/dev/null
  STARTED_SESSION_RUN=1
fi

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
  "$PYTHON_BIN" "$ROOT_DIR/scripts/index_autoresearch_run.py" "$RESULTS_JSON" --index_dir "$INDEX_DIR"
  if [[ "$USE_SESSION_STATE" == "1" && "$STARTED_SESSION_RUN" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" finish-run --results_json "$RESULTS_JSON" >/dev/null
  fi
elif [[ "$USE_SESSION_STATE" == "1" && "$STARTED_SESSION_RUN" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" abort-run --run_id "$RUN_ID" --reason "missing_results_json" >/dev/null
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "Autoresearch experiment finished"
echo "Run results: $RESULTS_JSON"
echo "Run metrics: $OUTPUT_DIR/metrics.jsonl"
echo "TensorBoard logdir: $OUTPUT_DIR/tensorboard"
if [[ -f "$PREFLIGHT_DECISION_JSON" ]]; then
  echo "Preflight decision: $PREFLIGHT_DECISION_JSON"
fi
if [[ "$USE_SESSION_STATE" == "1" ]]; then
  echo "Autoresearch session: $SESSION_JSON"
fi
echo "Latest index: $INDEX_DIR/latest.json"
echo "Best index: $INDEX_DIR/best.json"
