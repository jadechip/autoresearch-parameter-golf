#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export NPROC_PER_NODE=8
export CONFIG_JSON="${CONFIG_JSON:-$ROOT_DIR/configs/runpod_h100_8x_10min.json}"

exec bash "$ROOT_DIR/scripts/run_h100_eval.sh" "$@"
