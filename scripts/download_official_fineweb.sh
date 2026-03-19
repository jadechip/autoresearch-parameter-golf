#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
WITH_DOCS="${WITH_DOCS:-0}"

echo "Downloading official Parameter Golf FineWeb cache"
echo "VARIANT=$VARIANT"
echo "TRAIN_SHARDS=$TRAIN_SHARDS"
echo "WITH_DOCS=$WITH_DOCS"

uv sync --extra dev --extra tokenizer

CMD=(
  uv run python prepare.py official-fineweb
  --variant "$VARIANT"
  --train-shards "$TRAIN_SHARDS"
)

if [[ "$WITH_DOCS" == "1" ]]; then
  CMD+=(--with-docs)
fi

"${CMD[@]}"
