#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
WITH_DOCS="${WITH_DOCS:-0}"

echo "Downloading official Parameter Golf FineWeb cache"
echo "VARIANT=$VARIANT"
echo "TRAIN_SHARDS=$TRAIN_SHARDS"
echo "WITH_DOCS=$WITH_DOCS"

uv sync --extra dev --extra tokenizer

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

CMD=(
  "$PYTHON_BIN" prepare.py official-fineweb
  --variant "$VARIANT"
  --train-shards "$TRAIN_SHARDS"
)

if [[ "$WITH_DOCS" == "1" ]]; then
  CMD+=(--with-docs)
fi

"${CMD[@]}"
