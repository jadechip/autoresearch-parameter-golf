#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE_DIR="${STATE_DIR:-$ROOT_DIR/.minimal_autoresearch}"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$ROOT_DIR/runs/minimal_autoresearch_5090}"
RUN_LOG_DIR="$STATE_DIR/runs"
LOCK_DIR="$STATE_DIR/loop.lock"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/minimal_autoresearch/CODEX_MINIMAL_ONE_SHOT_PROMPT.md}"
DEFAULT_CONTEXT_FILE="$ROOT_DIR/minimal_autoresearch/parameter_golf_agent_context_capsule.md"
PROMPT_CONTEXT_FILES="${PROMPT_CONTEXT_FILES:-}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-}"
LAST_MESSAGE_PATH="$RUN_LOG_DIR/last-message.txt"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

resolve_context_file() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT_DIR/$path"
  fi
}

mkdir -p "$RUN_LOG_DIR"

if [[ -z "$PROMPT_CONTEXT_FILES" && -f "$DEFAULT_CONTEXT_FILE" ]]; then
  PROMPT_CONTEXT_FILES="$DEFAULT_CONTEXT_FILE"
fi

if [[ ! -f "$STATE_DIR/state.json" ]]; then
  echo "Missing minimal autoresearch state: $STATE_DIR/state.json" >&2
  echo "Run: .venv/bin/python minimal_autoresearch/state.py init --baseline_results runs/autoresearch_5090/index/best_raw.json" >&2
  exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing Codex prompt file: $PROMPT_FILE" >&2
  exit 2
fi

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
  echo "Another minimal autoresearch loop appears to be running: $LOCK_DIR" >&2
  exit 2
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

iter_ts="$(date -u +%Y%m%d-%H%M%S)"
iter_log="$RUN_LOG_DIR/codex-minimal-iteration-$iter_ts.log"
iter_last="$RUN_LOG_DIR/codex-minimal-iteration-$iter_ts.last.txt"
iter_prompt="$RUN_LOG_DIR/codex-minimal-iteration-$iter_ts.prompt.md"

export STATE_DIR
export AUTORESEARCH_ROOT

cp "$PROMPT_FILE" "$iter_prompt"

if [[ -n "$PROMPT_CONTEXT_FILES" ]]; then
  IFS=':' read -r -a context_files <<< "$PROMPT_CONTEXT_FILES"
  {
    printf '\n\n## Dynamically Injected Context\n\n'
    printf 'The following user-authored context files are injected at runtime for this iteration.\n'
    printf 'Treat them as additional priors and constraints for this run.\n'
    printf 'If they conflict with the frozen protocol or the actual repo state, follow the frozen protocol and repo state.\n'
  } >> "$iter_prompt"
  for context_file in "${context_files[@]}"; do
    [[ -z "$context_file" ]] && continue
    resolved_context_file="$(resolve_context_file "$context_file")"
    if [[ ! -f "$resolved_context_file" ]]; then
      echo "Missing prompt context file: $resolved_context_file" >&2
      exit 2
    fi
    {
      printf '\n\n### Begin %s\n\n' "${resolved_context_file#$ROOT_DIR/}"
      cat "$resolved_context_file"
      printf '\n\n### End %s\n' "${resolved_context_file#$ROOT_DIR/}"
    } >> "$iter_prompt"
  done
fi

cmd=("$CODEX_BIN" exec --dangerously-bypass-approvals-and-sandbox -C "$ROOT_DIR" --color never --output-last-message "$iter_last")
if [[ -n "$CODEX_MODEL" ]]; then
  cmd+=(-m "$CODEX_MODEL")
fi
cmd+=(-)

printf '%s minimal_iteration_start log=%s prompt=%s\n' "$(timestamp)" "$iter_log" "$iter_prompt"
set +e
"${cmd[@]}" < "$iter_prompt" >"$iter_log" 2>&1
status=$?
set -e

if [[ $status -ne 0 ]]; then
  printf '%s minimal_iteration_failed exit_code=%s log=%s\n' "$(timestamp)" "$status" "$iter_log" >&2
  exit $status
fi

if [[ -f "$iter_last" ]]; then
  cp "$iter_last" "$LAST_MESSAGE_PATH"
  printf '%s minimal_iteration_complete log=%s last_message=%s\n' "$(timestamp)" "$iter_log" "$iter_last"
else
  printf '%s minimal_iteration_complete log=%s\n' "$(timestamp)" "$iter_log"
fi
