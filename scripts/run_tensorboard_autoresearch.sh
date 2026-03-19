#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOGDIR="${LOGDIR:-$ROOT_DIR/runs/autoresearch_5090/runs}"
PORT="${PORT:-6006}"

echo "Starting TensorBoard"
echo "LOGDIR=$LOGDIR"
echo "PORT=$PORT"
echo "Open the forwarded URL in your browser after exposing port $PORT from the remote host."

if [[ -x "$ROOT_DIR/.venv/bin/tensorboard" ]]; then
  exec "$ROOT_DIR/.venv/bin/tensorboard" --logdir "$LOGDIR" --bind_all --port "$PORT"
fi

exec uv run tensorboard --logdir "$LOGDIR" --bind_all --port "$PORT"
