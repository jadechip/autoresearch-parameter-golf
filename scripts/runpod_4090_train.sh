#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_JSON="${CONFIG_JSON:-$ROOT_DIR/configs/runpod_4090_single_gpu.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/runpod_4090_single_gpu}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$ROOT_DIR/results.tsv}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
TRAIN_PATTERN="${TRAIN_PATTERN:-$ROOT_DIR/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-$ROOT_DIR/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"

if ! compgen -G "$TRAIN_PATTERN" > /dev/null; then
  echo "No training shards matched TRAIN_PATTERN=$TRAIN_PATTERN" >&2
  echo "Run: TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh" >&2
  echo "For a smaller iteration set: TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh" >&2
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

echo "Starting training"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TRAIN_PATTERN=$TRAIN_PATTERN"
echo "VAL_PATTERN=$VAL_PATTERN"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"

CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUBLAS_WORKSPACE_CONFIG
export PYTHONUNBUFFERED=1

uv sync --frozen --extra dev --extra tokenizer

if ! uv run python train.py \
  --config_json "$CONFIG_JSON" \
  --train_pattern "$TRAIN_PATTERN" \
  --val_pattern "$VAL_PATTERN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --results_tsv_path "$RESULTS_TSV_PATH" \
  --max_wallclock_seconds "$MAX_WALLCLOCK_SECONDS" \
  "$@"; then
  echo "Training failed" >&2
  if [[ -f "$OUTPUT_DIR/crash.json" ]]; then
    echo "Crash summary:" >&2
    cat "$OUTPUT_DIR/crash.json" >&2
  fi
  exit 1
fi

echo "Training finished"
echo "Results: $OUTPUT_DIR/results.json"
