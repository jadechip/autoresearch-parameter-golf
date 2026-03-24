#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.autoresearch}"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$ROOT_DIR/runs/autoresearch_5090}"
SESSION_JSON="$STATE_DIR/session.json"
ACTIVITY_LOG="$STATE_DIR/activity.log"
ERRORS_LOG="$STATE_DIR/errors.log"
RUN_LOG_DIR="$STATE_DIR/runs"
LOCK_DIR="$STATE_DIR/loop.lock"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
CONTINUE_ON_AGENT_FAILURE="${CONTINUE_ON_AGENT_FAILURE:-0}"

usage() {
  cat <<'EOF2' >&2
Usage: bash scripts/run_codex_autoresearch_loop.sh [--iterations N] [--sleep-seconds N] [--prompt-file PATH] [--model MODEL] [--continue-on-agent-failure]
EOF2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      [[ $# -ge 2 ]] || usage
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --sleep-seconds)
      [[ $# -ge 2 ]] || usage
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -ge 2 ]] || usage
      PROMPT_FILE="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || usage
      CODEX_MODEL="$2"
      shift 2
      ;;
    --continue-on-agent-failure)
      CONTINUE_ON_AGENT_FAILURE=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_activity() {
  local line
  line="$(timestamp) $*"
  printf '%s\n' "$line" | tee -a "$ACTIVITY_LOG"
}

log_error() {
  local line
  line="$(timestamp) $*"
  printf '%s\n' "$line" | tee -a "$ERRORS_LOG" >&2
}

ensure_ready() {
  while true; do
    if "$PYTHON_BIN" "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" require-ready >/dev/null 2>&1; then
      return 0
    fi
    if [[ "$WAIT_FOR_READY" != "1" ]]; then
      return 1
    fi
    log_activity "session_not_ready sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
  done
}

mkdir -p "$RUN_LOG_DIR"
touch "$ACTIVITY_LOG" "$ERRORS_LOG"

if [[ ! -f "$SESSION_JSON" ]]; then
  echo "Missing autoresearch session: $SESSION_JSON" >&2
  echo "Run: bash scripts/init_autoresearch_session.sh" >&2
  exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing Codex prompt file: $PROMPT_FILE" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN" >&2
  echo "Run: bash scripts/bootstrap.sh" >&2
  exit 2
fi

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export STATE_DIR
export AUTORESEARCH_ROOT

if [[ "$CODEX_BIN" == */* ]]; then
  if [[ ! -x "$CODEX_BIN" ]]; then
    echo "Codex binary not executable: $CODEX_BIN" >&2
    exit 2
  fi
elif ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "Codex binary not found: $CODEX_BIN" >&2
  exit 2
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another Codex autoresearch supervisor appears to be running: $LOCK_DIR" >&2
  exit 2
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

ensure_ready

log_activity "codex_loop_start prompt_file=$PROMPT_FILE max_iterations=$MAX_ITERATIONS sleep_seconds=$SLEEP_SECONDS autoresearch_root=$AUTORESEARCH_ROOT"

iteration=0
while true; do
  if [[ "$MAX_ITERATIONS" != "0" && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    log_activity "codex_loop_complete iterations=$iteration"
    break
  fi

  ensure_ready

  iteration=$((iteration + 1))
  iter_ts="$(date -u +%Y%m%d-%H%M%S)"
  iter_log="$RUN_LOG_DIR/codex-iteration-$iter_ts.log"
  iter_last="$RUN_LOG_DIR/codex-iteration-$iter_ts.last.txt"

  session_payload="$($PYTHON_BIN "$ROOT_DIR/scripts/autoresearch_state.py" --state_dir "$STATE_DIR" show)"
  lane="$(printf '%s' "$session_payload" | "$PYTHON_BIN" -c 'import json,sys; print(((json.load(sys.stdin).get("search_policy") or {}).get("lane")) or "-")')"
  accepted_run_id="$(printf '%s' "$session_payload" | "$PYTHON_BIN" -c 'import json,sys; print((json.load(sys.stdin).get("accepted_run_id")) or "-")')"

  cmd=("$CODEX_BIN" exec --dangerously-bypass-approvals-and-sandbox -C "$ROOT_DIR" --color never --output-last-message "$iter_last")
  if [[ -n "$CODEX_MODEL" ]]; then
    cmd+=(-m "$CODEX_MODEL")
  fi
  cmd+=(-)

  log_activity "iteration_start iteration=$iteration lane=$lane accepted_run_id=$accepted_run_id log=$iter_log"
  set +e
  "${cmd[@]}" < "$PROMPT_FILE" >"$iter_log" 2>&1
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    log_error "iteration_failed iteration=$iteration lane=$lane accepted_run_id=$accepted_run_id exit_code=$status log=$iter_log"
    if [[ "$CONTINUE_ON_AGENT_FAILURE" == "1" ]]; then
      sleep "$SLEEP_SECONDS"
      continue
    fi
    exit $status
  fi

  if [[ -f "$iter_last" ]]; then
    log_activity "iteration_complete iteration=$iteration lane=$lane accepted_run_id=$accepted_run_id log=$iter_log last_message=$iter_last"
  else
    log_activity "iteration_complete iteration=$iteration lane=$lane accepted_run_id=$accepted_run_id log=$iter_log"
  fi

  sleep "$SLEEP_SECONDS"
done
