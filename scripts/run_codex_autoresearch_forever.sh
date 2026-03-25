#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"

LANE="${LANE:-frontier_pure_model}"
AGGR_CHUNK_ITERS="${AGGR_CHUNK_ITERS:-6}"
REFINE_CHUNK_ITERS="${REFINE_CHUNK_ITERS:-4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
TRIES_PER_IDEA="${TRIES_PER_IDEA:-6}"

if [[ ! -f ./.autoresearch/session.json ]]; then
  LANE="$LANE" FORCE=1 bash scripts/init_autoresearch_session.sh
fi

if [[ ! -f ./.autoresearch_aggressive/session.json || ! -f ./.autoresearch_aggressive/aggressive_campaign.json ]]; then
  LANE="$LANE" FORCE=1 TRIES_PER_IDEA="$TRIES_PER_IDEA" bash scripts/init_aggressive_autoresearch_session.sh
fi

while true; do
  campaign_status="$(
    "$PYTHON_BIN" scripts/aggressive_autoresearch_campaign.py --state_dir ./.autoresearch_aggressive show \
      | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])'
  )"

  if [[ "$campaign_status" == "completed" ]]; then
    "$PYTHON_BIN" scripts/aggressive_autoresearch_campaign.py \
      --state_dir ./.autoresearch_aggressive \
      init \
      --ideas_json ./configs/aggressive_autoresearch_ideas.json \
      --tries_per_idea "$TRIES_PER_IDEA" \
      --force >/dev/null
  fi

  bash scripts/run_codex_autoresearch_aggressive_loop.sh \
    --keep-going-after-contender \
    --iterations "$AGGR_CHUNK_ITERS" \
    --sleep-seconds "$SLEEP_SECONDS"

  "$PYTHON_BIN" scripts/autoresearch_state.py \
    --state_dir ./.autoresearch \
    init-from-tracked \
    --tracked_state ./state/autoresearch/accepted_state.json \
    --lane "$LANE" \
    --force >/dev/null

  bash scripts/run_codex_autoresearch_loop.sh \
    --iterations "$REFINE_CHUNK_ITERS" \
    --sleep-seconds "$SLEEP_SECONDS"

  "$PYTHON_BIN" scripts/autoresearch_state.py \
    --state_dir ./.autoresearch \
    sync-tracked-accepted >/dev/null

  "$PYTHON_BIN" scripts/autoresearch_state.py \
    --state_dir ./.autoresearch_aggressive \
    init-from-tracked \
    --tracked_state ./state/autoresearch/accepted_state.json \
    --lane "$LANE" \
    --force >/dev/null
done
