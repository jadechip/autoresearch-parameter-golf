#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch_aggressive}"
TRACKED_ACCEPTED_STATE="${TRACKED_ACCEPTED_STATE:-$ROOT_DIR/state/autoresearch/accepted_state.json}"
MAIN_STATE_DIR="${MAIN_STATE_DIR:-$ROOT_DIR/.autoresearch}"
IDEAS_JSON="${IDEAS_JSON:-$ROOT_DIR/configs/aggressive_autoresearch_ideas.json}"
TRIES_PER_IDEA="${TRIES_PER_IDEA:-6}"
FORCE="${FORCE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

if [[ ! -f "$TRACKED_ACCEPTED_STATE" ]]; then
  if [[ -f "$MAIN_STATE_DIR/session.json" ]]; then
    echo "Tracked accepted state missing; materializing it from $MAIN_STATE_DIR"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$MAIN_STATE_DIR" sync-tracked-accepted >/dev/null
  fi
fi

if [[ ! -f "$TRACKED_ACCEPTED_STATE" ]]; then
  echo "Missing tracked accepted state: $TRACKED_ACCEPTED_STATE" >&2
  echo "Either restore state/autoresearch/accepted_state.json or keep a valid main session at $MAIN_STATE_DIR." >&2
  exit 2
fi

if [[ ! -f "$IDEAS_JSON" ]]; then
  echo "Missing aggressive ideas file: $IDEAS_JSON" >&2
  exit 2
fi

echo "Initializing aggressive autoresearch session"
echo "STATE_DIR=$STATE_DIR"
echo "TRACKED_ACCEPTED_STATE=$TRACKED_ACCEPTED_STATE"
echo "IDEAS_JSON=$IDEAS_JSON"
echo "TRIES_PER_IDEA=$TRIES_PER_IDEA"

tracked_cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py"
  --state_dir "$STATE_DIR"
  init-from-tracked
  --tracked_state "$TRACKED_ACCEPTED_STATE"
)
campaign_cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/aggressive_autoresearch_campaign.py"
  --state_dir "$STATE_DIR"
  init
  --ideas_json "$IDEAS_JSON"
  --tries_per_idea "$TRIES_PER_IDEA"
)

if [[ "$FORCE" == "1" ]]; then
  tracked_cmd+=(--force)
  campaign_cmd+=(--force)
fi

"${tracked_cmd[@]}"
"${campaign_cmd[@]}"

echo "Aggressive autoresearch session ready"
echo "Session: $STATE_DIR/session.json"
echo "Campaign: $STATE_DIR/aggressive_campaign.json"
