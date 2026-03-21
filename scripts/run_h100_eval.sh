#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NPROC_PER_NODE" == "8" ]]; then
  DEFAULT_CONFIG_JSON="$ROOT_DIR/configs/runpod_h100_8x_10min.json"
  DEFAULT_RUN_ROOT="$ROOT_DIR/runs/runpod_h100_8x_eval"
else
  DEFAULT_CONFIG_JSON="$ROOT_DIR/configs/runpod_h100_1x_10min.json"
  DEFAULT_RUN_ROOT="$ROOT_DIR/runs/runpod_h100_1x_eval"
fi

if [[ -z "${ARTIFACT_PATH:-}" ]]; then
  echo "ARTIFACT_PATH must be set to a submission_bundle directory, manifest.json, or model blob." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-h100-eval-${NPROC_PER_NODE}x-${STAMP}}"
CONFIG_JSON="${CONFIG_JSON:-$DEFAULT_CONFIG_JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_RUN_ROOT/$RUN_ID}"
RESULTS_TSV_PATH="${RESULTS_TSV_PATH:-$DEFAULT_RUN_ROOT/results.tsv}"
VAL_PATTERN="${VAL_PATTERN:-$ROOT_DIR/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
EVAL_LOG_PATH="${EVAL_LOG_PATH:-$OUTPUT_DIR/eval.log}"

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

echo "Starting H100 artifact evaluation"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_JSON=$CONFIG_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESULTS_TSV_PATH=$RESULTS_TSV_PATH"
echo "ARTIFACT_PATH=$ARTIFACT_PATH"
echo "EVAL_LOG_PATH=$EVAL_LOG_PATH"
echo "VAL_PATTERN=$VAL_PATTERN"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"

CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUBLAS_WORKSPACE_CONFIG
export PYTHONUNBUFFERED=1

uv sync --frozen --extra dev --extra tokenizer

set +e
uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
  --config_json "$CONFIG_JSON" \
  --val_pattern "$VAL_PATTERN" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --results_tsv_path "$RESULTS_TSV_PATH" \
  --load_artifact_path "$ARTIFACT_PATH" \
  --evaluate_only \
  --run_name "$RUN_ID" \
  "$@" 2>&1 | tee "$EVAL_LOG_PATH"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Evaluation failed" >&2
  if [[ -f "$OUTPUT_DIR/crash.json" ]]; then
    echo "Crash summary:" >&2
    cat "$OUTPUT_DIR/crash.json" >&2
  fi
  exit "$status"
fi

echo "Evaluation finished"
echo "Results: $OUTPUT_DIR/results.json"
echo "Eval log: $EVAL_LOG_PATH"
echo "Metrics: $OUTPUT_DIR/metrics.jsonl"
echo "TensorBoard logdir: $OUTPUT_DIR/tensorboard"
