#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
iteration=0

while true; do
  if [[ "$MAX_ITERATIONS" != "0" && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    printf '%s minimal_loop_complete iterations=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$iteration"
    break
  fi
  iteration=$((iteration + 1))
  printf '%s minimal_loop_iteration iteration=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$iteration"
  bash "$ROOT_DIR/minimal_autoresearch/run_once.sh"
  sleep "$SLEEP_SECONDS"
done
