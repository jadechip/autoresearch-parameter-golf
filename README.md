# autoresearch-parameter-golf

`autoresearch-parameter-golf` is a starter kit for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) contest.

It is not the official contest repo. Instead, it packages a strong recurrent-QAT baseline into a repo that is easier to train, export, reload, evaluate, and search with an external coding agent. The main goal is to give you a practical boilerplate that:

- follows the Parameter Golf data and artifact conventions closely
- keeps `train.py` as the main editable surface
- works standalone for manual runs
- ships with an `autoresearch`-style loop out of the box
- writes structured outputs so agents do not need to scrape logs

If you want the official challenge rules, dataset notes, and submission requirements, start with:

- `https://github.com/openai/parameter-golf`

## What This Repo Is

This repo takes the original single-file reference implementation and turns it into a more usable experimentation harness while preserving the intended baseline architecture by default:

- decoder-only LM
- tied embeddings
- grouped-query attention
- RoPE
- RMSNorm
- ReLU^2 MLP
- depth recurrence via shared blocks
- per-loop LoRA-style adapters
- fake INT8 quantization during training
- row-wise INT8 export with zlib packing
- optional linear weight averaging

The core files are:

- `train.py`: main mutation target for model/training/search changes
- `prepare.py`: fixed smoke-data, tokenizer, and artifact-eval utilities
- `program.md`: rules for external coding agents
- `CODEX_AUTORESEARCH_PROMPT.md`: copy/paste prompt for Codex
- `validate_results.py`: fixed `results.json` schema validator
- `summarize_artifact.py`: submission-oriented artifact summary
- `watch_run.py`: live terminal monitor for one run
- `compare_runs.py`: compact terminal comparison view for many runs
- `train_pgolf_recurrent_qat.py`: compatibility wrapper for the old entrypoint

## What This Repo Is Not

- not the official leaderboard repo
- not a fully code-golfed final submission
- not locked to one training workflow

Think of it as a contest-oriented starter kit: a clean place to build, measure, and search before final submission packaging.

## The Fastest Way To Get Going

If you just want to get to a real run quickly:

```bash
bash scripts/bootstrap.sh
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
bash scripts/run_autoresearch_experiment.sh
```

Then, in a second terminal:

```bash
uv run pgolf-watch-run ./runs/autoresearch_5090/index/latest.json
```

That gives you:

- official cached FineWeb shards in the local `./data/` layout
- one fixed-budget training run
- a live view of loss, LR, validation, and timing from `metrics.jsonl`

## Installation

Use a clean environment. The repo is pinned in `pyproject.toml`.

Recommended:

```bash
bash scripts/bootstrap.sh
```

Manual equivalent:

```bash
uv sync --frozen --extra dev --extra tokenizer
```

If you do not need real SentencePiece evaluation:

```bash
bash scripts/bootstrap.sh --no-tokenizer
```

There is also a small `Makefile` for common commands:

```bash
make install
make smoke
make train-5090
make autoresearch-baseline
make watch-latest
make compare-autoresearch
```

## Quick Start Paths

### 1. Smoke Test The Full Pipeline

Create tiny synthetic shards:

```bash
uv run pgolf-prepare smoke-data --output_dir ./smoke_data
```

Run a short smoke experiment:

```bash
uv run pgolf-train --config_json ./smoke_data/smoke_config.json --max_wallclock_seconds 30
```

Validate the structured output:

```bash
uv run pgolf-validate-results ./smoke_data/run/results.json
```

Reload and evaluate the exported artifact:

```bash
uv run pgolf-prepare eval-artifact \
  --config_json ./smoke_data/smoke_config.json \
  --artifact_path ./smoke_data/run/submission_bundle \
  --output_dir ./smoke_data/eval
```

Summarize submission-relevant bytes:

```bash
uv run pgolf-summarize-artifact --results_json ./smoke_data/run/results.json
```

### 2. Run A Real Single-GPU Training Job

Download the official cached challenge data:

```bash
TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh
```

For a smaller iteration subset:

```bash
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
```

This mirrors the official `openai/parameter-golf` data layout under:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

Run the conservative 5090 preset:

```bash
bash scripts/runpod_5090_train.sh
```

Watch it live:

```bash
uv run pgolf-watch-run ./runs/runpod_5090_single_gpu
```

### 3. Start The Built-In Autoresearch Loop

For search on cheaper hardware, use the fixed-budget 5090 loop:

```bash
bash scripts/run_autoresearch_experiment.sh
```

This creates:

- one run directory under `./runs/autoresearch_5090/runs/`
- append-only `./runs/autoresearch_5090/results.tsv`
- stable agent-readable pointers:
  - `./runs/autoresearch_5090/index/latest.json`
  - `./runs/autoresearch_5090/index/best.json`

Watch the latest run:

```bash
make watch-latest
```

Compare completed runs:

```bash
make compare-autoresearch
```

For Codex sessions, use:

- `CODEX_AUTORESEARCH_PROMPT.md`

For the durable policy and allowed search space, use:

- `program.md`

## How This Relates To Autoresearch

This repo is shaped to work well with `karpathy/autoresearch`-style loops:

- one main editable file: `train.py`
- fixed helper utilities: `prepare.py`, validators, wrappers
- structured results instead of raw-log control
- repeatable fixed-budget experiment wrapper
- branch/history-based search policy in `program.md`

The intended workflow is:

1. search cheaply on `1x5090`
2. promote only meaningful winners to `1xH100`
3. reserve `8xH100` for exact-budget rehearsal and serious submission attempts

That keeps the cheap search tier aligned with the real contest without burning expensive cluster time on bad ideas.

## Structured Outputs

Do not control runs by tailing raw logs. Consume these files instead:

- `output_dir/results.json`: fixed-schema result summary
- `output_dir/crash.json`: structured crash metadata on failure
- `output_dir/metrics.jsonl`: append-only per-step/per-eval metrics stream for live monitoring
- `results.tsv`: append-only experiment table
- `output_dir/submission_bundle/manifest.json`: export manifest and byte accounting
- `output_dir/checkpoints/latest.pt` and `final.pt`: resume points

The training script also prints grep-friendly final lines:

- `val_bpb:`
- `training_seconds:`
- `total_seconds:`
- `peak_vram_mb:`
- `mfu_percent:`
- `total_tokens_M:`
- `num_steps:`
- `num_params_M:`
- `depth:`
- `artifact_bytes:`

## Live Monitoring

Watch one run, including continuously updating LR/loss charts:

```bash
uv run pgolf-watch-run ./runs/autoresearch_5090/index/latest.json
```

Choose different metrics:

```bash
uv run pgolf-watch-run ./runs/runpod_5090_single_gpu --metrics train_loss,matrix_lr,val_bpb
```

Compare completed runs from `results.tsv`:

```bash
uv run pgolf-compare-runs --results_tsv ./runs/autoresearch_5090/results.tsv
```

## Notes On Artifact Accounting

- `train.py` is intentionally the main editable file for external search loops.
- `prepare.py`, `validate_results.py`, `summarize_artifact.py`, and the docs are intended to stay stable.
- Export accounting snapshots the counted code files into `submission_bundle/code/` and counts those exact bytes plus the compressed model bytes.
- The default counted code set is just `train.py`, which keeps artifact accounting close to Parameter Golf’s preference for a single counted training file.
- Autonomous runs should be sandboxed and should read `results.json` / `crash.json`, not arbitrary stdout.

## Tests

CPU-only:

```bash
pytest -q
```

Optional marker runs:

```bash
pytest -q -m cuda
pytest -q -m ddp
pytest -q -m compile_smoke
pytest -q -m sentencepiece_real
```
