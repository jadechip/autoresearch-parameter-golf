#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
TORCHRUN_BIN="$ROOT_DIR/.venv/bin/torchrun"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NPROC_PER_NODE" == "8" ]]; then
  DEFAULT_CONFIG_JSON="$ROOT_DIR/configs/runpod_h100_8x_10min.json"
  DEFAULT_RUN_ROOT="$ROOT_DIR/runs/runpod_h100_8x_10min"
else
  DEFAULT_CONFIG_JSON="$ROOT_DIR/configs/runpod_h100_1x_10min.json"
  DEFAULT_RUN_ROOT="$ROOT_DIR/runs/runpod_h100_1x_10min"
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-h100-${NPROC_PER_NODE}x-${STAMP}}"
CONFIG_JSON="${CONFIG_JSON:-$DEFAULT_CONFIG_JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_RUN_ROOT/$RUN_ID}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$DEFAULT_RUN_ROOT/results.tsv}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_PATTERN="${TRAIN_PATTERN:-$ROOT_DIR/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-$ROOT_DIR/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-$OUTPUT_DIR/train.log}"

if ! compgen -G "$TRAIN_PATTERN" > /dev/null; then
  echo "No training shards matched TRAIN_PATTERN=$TRAIN_PATTERN" >&2
  echo "Run: TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh" >&2
  exit 2
fi

if ! compgen -G "$VAL_PATTERN" > /dev/null; then
  echo "No validation shards matched VAL_PATTERN=$VAL_PATTERN" >&2
  echo "Run: TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh" >&2
  exit 2
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "Tokenizer file does not exist: TOKENIZER_PATH=$TOKENIZER_PATH" >&2
  echo "Run: TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$RESULTS_TSV_PATH")"

echo "Starting H100 training run"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESULTS_TSV_PATH=$RESULTS_TSV_PATH"
echo "TRAIN_LOG_PATH=$TRAIN_LOG_PATH"
echo "TRAIN_PATTERN=$TRAIN_PATTERN"
echo "VAL_PATTERN=$VAL_PATTERN"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"

CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUBLAS_WORKSPACE_CONFIG
export PYTHONUNBUFFERED=1

uv sync --frozen --extra dev --extra tokenizer

if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "Missing virtualenv torchrun: $TORCHRUN_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

set +e
"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
  --config_json "$CONFIG_JSON" \
  --train_pattern "$TRAIN_PATTERN" \
  --val_pattern "$VAL_PATTERN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --results_tsv_path "$RESULTS_TSV_PATH" \
  --max_wallclock_seconds "$MAX_WALLCLOCK_SECONDS" \
  --run_name "$RUN_ID" \
  "$@" 2>&1 | tee "$TRAIN_LOG_PATH"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Training failed" >&2
  if [[ -f "$OUTPUT_DIR/crash.json" ]]; then
    echo "Crash summary:" >&2
    cat "$OUTPUT_DIR/crash.json" >&2
  fi
  exit "$status"
fi

echo "Training finished"
echo "Results: $OUTPUT_DIR/results.json"
echo "Train log: $TRAIN_LOG_PATH"
echo "Metrics: $OUTPUT_DIR/metrics.jsonl"
echo "TensorBoard logdir: $OUTPUT_DIR/tensorboard"
echo "Artifact bundle: $OUTPUT_DIR/submission_bundle"
