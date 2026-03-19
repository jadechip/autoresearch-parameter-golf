#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_DEV=1
WITH_TOKENIZER=1

for arg in "$@"; do
  case "$arg" in
    --no-dev)
      WITH_DEV=0
      ;;
    --no-tokenizer)
      WITH_TOKENIZER=0
      ;;
    *)
      echo "Usage: bash scripts/bootstrap.sh [--no-dev] [--no-tokenizer]" >&2
      exit 2
      ;;
  esac
done

extras=()
if [[ "$WITH_DEV" == "1" ]]; then
  extras+=(--extra dev)
fi
if [[ "$WITH_TOKENIZER" == "1" ]]; then
  extras+=(--extra tokenizer)
fi

echo "Bootstrapping autoresearch-parameter-golf"
echo "ROOT_DIR=$ROOT_DIR"
echo "WITH_DEV=$WITH_DEV"
echo "WITH_TOKENIZER=$WITH_TOKENIZER"

uv sync --frozen "${extras[@]}"

echo
echo "Installed environment."
echo "Useful next commands:"
echo "  make smoke"
echo "  TRAIN_SHARDS=1 make download-data"
echo "  make train-5090"
echo "  make autoresearch-baseline"
echo "  make monitor-latest"
echo "  make compare-autoresearch"
