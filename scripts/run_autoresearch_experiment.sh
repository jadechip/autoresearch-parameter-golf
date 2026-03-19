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

mkdir -p "$RUNS_DIR" "$INDEX_DIR"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another autoresearch experiment appears to be running: $LOCK_DIR" >&2
  exit 2
fi

cleanup() {
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
  "crash_path": "$OUTPUT_DIR/crash.json"
}
EOF

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
fi

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "Autoresearch experiment finished"
echo "Run results: $RESULTS_JSON"
echo "Run metrics: $OUTPUT_DIR/metrics.jsonl"
echo "Latest index: $INDEX_DIR/latest.json"
echo "Best index: $INDEX_DIR/best.json"
