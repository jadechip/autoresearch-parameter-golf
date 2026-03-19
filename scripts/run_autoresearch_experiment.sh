#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$ROOT_DIR/runs/autoresearch_5090}"
RUNS_DIR="${RUNS_DIR:-$AUTORESEARCH_ROOT/runs}"
INDEX_DIR="${INDEX_DIR:-$AUTORESEARCH_ROOT/index}"
CONFIG_JSON="${CONFIG_JSON:-$ROOT_DIR/configs/autoresearch_5090_5min.json}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$AUTORESEARCH_ROOT/results.tsv}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
RUN_ID="${RUN_ID:-ar5090-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/$RUN_ID}"
LOCK_DIR="$AUTORESEARCH_ROOT/.lock"
ACTIVE_JSON="$INDEX_DIR/active.json"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch}"
SESSION_JSON="$STATE_DIR/session.json"
ALLOW_UNINITIALIZED_SESSION="${ALLOW_UNINITIALIZED_SESSION:-0}"

mkdir -p "$RUNS_DIR" "$INDEX_DIR"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

USE_SESSION_STATE=0
if [[ -f "$SESSION_JSON" ]]; then
  uv run python "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" require-ready >/dev/null
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
  if [[ "$USE_SESSION_STATE" == "1" && ! -f "$OUTPUT_DIR/results.json" ]]; then
    uv run python "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" abort-run --run_id "$RUN_ID" --reason "wrapper_cleanup" >/dev/null 2>&1 || true
  fi
  rm -f "$ACTIVE_JSON"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting autoresearch experiment"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "INDEX_DIR=$INDEX_DIR"
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"

cat > "$ACTIVE_JSON" <<EOF
{
  "status": "running",
  "run_id": "$RUN_ID",
  "output_dir": "$OUTPUT_DIR",
  "results_path": "$OUTPUT_DIR/results.json",
  "metrics_path": "$OUTPUT_DIR/metrics.jsonl",
  "tensorboard_log_dir": "$OUTPUT_DIR/tensorboard",
  "crash_path": "$OUTPUT_DIR/crash.json"
}
EOF

if [[ "$USE_SESSION_STATE" == "1" ]]; then
  uv run python "$ROOT_DIR/scripts/autoresearch_state.py" \
    --state_dir "$STATE_DIR" \
    start-run \
    --run_id "$RUN_ID" \
    --output_dir "$OUTPUT_DIR" \
    --results_path "$OUTPUT_DIR/results.json" \
    --metrics_path "$OUTPUT_DIR/metrics.jsonl" \
    --tensorboard_log_dir "$OUTPUT_DIR/tensorboard" \
    --crash_path "$OUTPUT_DIR/crash.json" >/dev/null
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
  uv run python "$ROOT_DIR/scripts/index_autoresearch_run.py" "$RESULTS_JSON" --index_dir "$INDEX_DIR"
  if [[ "$USE_SESSION_STATE" == "1" ]]; then
    uv run python "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" finish-run --results_json "$RESULTS_JSON" >/dev/null
  fi
elif [[ "$USE_SESSION_STATE" == "1" ]]; then
  uv run python "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" abort-run --run_id "$RUN_ID" --reason "missing_results_json" >/dev/null
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "Autoresearch experiment finished"
echo "Run results: $RESULTS_JSON"
echo "Run metrics: $OUTPUT_DIR/metrics.jsonl"
echo "TensorBoard logdir: $OUTPUT_DIR/tensorboard"
if [[ "$USE_SESSION_STATE" == "1" ]]; then
  echo "Autoresearch session: $SESSION_JSON"
fi
echo "Latest index: $INDEX_DIR/latest.json"
echo "Best index: $INDEX_DIR/best.json"
