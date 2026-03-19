# autoresearch-parameter-golf

`autoresearch-parameter-golf` is a starter kit for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) contest.

It is not the official contest repo. It is a practical training and experimentation repo built around the same challenge data format and artifact constraints, with `autoresearch`-style Codex iteration set up out of the box.

The intended use is:

1. run short fixed-budget search loops on a cheaper GPU such as `1x RTX 5090`
2. let Codex mutate `train.py` inside a controlled autoresearch loop
3. promote only the best candidates to more expensive `H100` and `8xH100` production rehearsals

## Core Workflow

This is the main path the README is optimized for.

### 1. Bootstrap The Repo

```bash
bash scripts/bootstrap.sh
```

### 2. Create An Autoresearch Branch

```bash
git checkout -b autoresearch/<tag>
```

### 3. Download The Official Parameter Golf Data

Use the official cached FineWeb layout mirrored from the upstream contest repo:

```bash
TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh
```

That writes the expected local files under:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

### 4. Run One Baseline Autoresearch Experiment

```bash
RUN_ID=baseline_5090_5min bash scripts/run_autoresearch_experiment.sh
```

This creates:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`
- `./runs/autoresearch_5090/results.tsv`

### 5. Watch The Latest Run In Another Terminal

```bash
uv run pgolf-watch-run ./runs/autoresearch_5090/index/latest.json
```

Or compare completed runs:

```bash
uv run pgolf-compare-runs --results_tsv ./runs/autoresearch_5090/results.tsv
```

### 6. Start Codex In The Repo Root

Codex should run on the same machine as the training job so it can edit the repo and launch experiments locally.

Start Codex in this directory and use:

- `CODEX_AUTORESEARCH_PROMPT.md` for the bootstrap prompt
- `program.md` for the durable search policy

The expected Codex behavior is:

- edit `train.py`
- run one experiment at a time with `bash scripts/run_autoresearch_experiment.sh`
- read `latest.json` and `best.json`
- keep meaningful winners
- revert losers
- use git history as experiment memory

## Why This Repo Exists

The official `openai/parameter-golf` repo is the source of truth for:

- challenge rules
- allowed training/eval budgets
- official cached data layout
- submission requirements

This repo exists to make experimentation easier:

- `train.py` stays the main editable file
- the data download flow mirrors the official contest setup
- runs write structured outputs instead of requiring log scraping
- Codex/autoresearch usage is documented and ready to use
- export, reload, validation, and artifact accounting are already wired together

## Important Files

- `train.py`: main mutation target for Codex
- `prepare.py`: fixed prep and artifact-eval utilities
- `program.md`: agent contract
- `CODEX_AUTORESEARCH_PROMPT.md`: copy/paste Codex prompt
- `AUTORESEARCH_SETUP.md`: longer setup notes
- `scripts/run_autoresearch_experiment.sh`: one fixed-budget autoresearch run
- `scripts/runpod_5090_train.sh`: manual single-GPU training wrapper

## Structured Outputs

Use structured files, not raw logs:

- `output_dir/results.json`
- `output_dir/crash.json`
- `output_dir/metrics.jsonl`
- `results.tsv`
- `output_dir/submission_bundle/manifest.json`

The autoresearch loop should primarily read:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`

## Install And Helper Commands

Recommended install:

```bash
bash scripts/bootstrap.sh
```

Manual equivalent:

```bash
uv sync --frozen --extra dev --extra tokenizer
```

Useful helpers:

```bash
make autoresearch-baseline
make watch-latest
make compare-autoresearch
```

## Notes

- This repo is a starter kit, not a finished leaderboard submission.
- The default counted code path is just `train.py`, which keeps artifact accounting close to Parameter Golf expectations.
- The 5090 loop is a production-proxy search tier, not the final target itself.
- The final contest objective is still the official one from `openai/parameter-golf`.
