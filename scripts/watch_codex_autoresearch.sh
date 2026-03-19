#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/watch_codex_autoresearch.py" "$@"
fi

exec uv run python "$ROOT_DIR/scripts/watch_codex_autoresearch.py" "$@"
