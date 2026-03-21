# Autoresearch Parameter Golf

`Autoresearch Parameter Golf` is my ongoing experiment repo for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition.

It is not the official contest repo. It is a practical starter repo and working setup for my own training and experimentation against the same challenge data format and artifact constraints, with `autoresearch`-style Codex iteration set up out of the box.

The intended use is:

1. run short fixed-budget search loops on a cheaper GPU such as `1x RTX 5090`
2. let Codex mutate `train.py` inside a controlled autoresearch loop
3. promote only the best candidates to more expensive `H100` and `8xH100` production rehearsals

## Main Workflow

This is the primary path this repo is built for: official FineWeb data, a 5090 search tier, and Codex driving an autoresearch loop.

### 1. Bootstrap The Repo

```bash
bash scripts/bootstrap.sh
```

### 2. Download The Official Parameter Golf Data

Use the official cached FineWeb layout mirrored from the upstream contest repo:

```bash
TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh
```

That writes the expected local files under:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

### 3. Run One Baseline Autoresearch Experiment

```bash
RUN_ID=baseline_5090_5min bash scripts/run_autoresearch_experiment.sh
```

This creates:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`
- `./runs/autoresearch_5090/results.tsv`

### 4. Initialize The Autoresearch Session Gate

```bash
bash scripts/init_autoresearch_session.sh
```

This creates a lightweight Ralph-style state directory at:

- `./.autoresearch/session.json`
- `./.autoresearch/experiments.jsonl`
- `./.autoresearch/notes.md`

Once that file exists, non-baseline autoresearch runs are gated on `session.json` being in `ready` state. This prevents Codex from starting prematurely against an uninitialized repo.

The session file also carries the current search policy for Codex, including:

- a soft 5090 artifact target band of `7,000,000` to `12,000,000` bytes
- a `~0.001 val_bpb` meaningful-win threshold
- a limit of `3` consecutive losing micro-tunes before the next run should be structural

### 5. Watch Progress With TensorBoard In Another Terminal

```bash
bash scripts/run_tensorboard_autoresearch.sh
```

Or compare completed runs:

```bash
uv run pgolf-compare-runs --results_tsv ./runs/autoresearch_5090/results.tsv
```

For a remote host, expose or forward port `6006`, then open TensorBoard in your local browser. A typical SSH tunnel looks like:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<host>
```

TensorBoard will show:

- the current run updating in real time
- previous runs beside it for comparison
- train loss, validation bpb, learning rates, throughput, and timing
- per-run grouping from the autoresearch run directory layout

## Run With Codex

This is a Codex-first repo. The intended loop is that Codex edits `train.py`, launches one experiment, reads structured outputs, and iterates.

### 1. Preferred Overnight Mode: Fresh `codex exec` Iterations

For long-running autonomous search, prefer the supervisor loop:

```bash
bash scripts/run_codex_autoresearch_loop.sh
```

This is the closest match to Ralph's execution model:

- one fresh `codex exec` process per iteration
- one experiment per invocation
- on-disk `.autoresearch/` state as memory
- the outer shell loop keeps the search running overnight

Loop logs land in:

- `./.autoresearch/activity.log`
- `./.autoresearch/errors.log`
- `./.autoresearch/runs/`

For a read-only live tail that follows the newest iteration log:

```bash
bash scripts/watch_codex_autoresearch.sh
```

### 2. Interactive Mode: Start Codex In The Repo Root

Codex should run on the same machine as the training job so it can edit the repo and launch experiments locally.

From the repo root:

- start Codex in `/workspace/autoresearch-parameter-golf`
- let Codex create and manage its own dedicated autoresearch branch
- give it the bootstrap prompt from `CODEX_AUTORESEARCH_PROMPT.md`

Recommended launch command on a dedicated remote box you control:

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

The two files Codex should read first are:

- `CODEX_AUTORESEARCH_PROMPT.md` for the bootstrap prompt
- `program.md` for the durable search policy

### 3. What Codex Should Do

The expected Codex behavior is:

- edit `train.py`
- run one experiment at a time with `bash scripts/run_autoresearch_experiment.sh`
- read `latest.json` and `best.json`
- read `.autoresearch/session.json` and `.autoresearch/notes.md`
- keep meaningful winners
- revert losers
- use git history as experiment memory
- cover structural / byte-allocation directions early instead of only micro-tuning optimizer values

### 4. Exact Human Setup For Codex

On the 5090 host:

```bash
cd /workspace/autoresearch-parameter-golf
bash scripts/bootstrap.sh
TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh
RUN_ID=baseline_5090_5min bash scripts/run_autoresearch_experiment.sh
bash scripts/init_autoresearch_session.sh
```

Then in one terminal:

```bash
bash scripts/run_tensorboard_autoresearch.sh
```

For the preferred overnight loop:

```bash
bash scripts/run_codex_autoresearch_loop.sh
```

Or in a supervised interactive Codex terminal:

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

Then print the prompt file:

```bash
cat CODEX_AUTORESEARCH_PROMPT.md
```

Then paste the prompt into Codex.

The loop script uses `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md` and launches a fresh `codex exec` each iteration.

Interactive Codex should create and use its own timestamped branch for the session.

### 5. What Codex Should Read After Each Run

Codex should use structured files, not logs:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`
- `./runs/autoresearch_5090/results.tsv`
- `./runs/autoresearch_5090/runs/<run_id>/metrics.jsonl`
- `./.autoresearch/session.json`
- `./.autoresearch/experiments.jsonl`
- `./.autoresearch/notes.md`

`latest.json` is the current run/result pointer. `best.json` is the standing best candidate.
`.autoresearch/session.json` is the readiness gate and accepted-state ledger.

The recommended live monitor is TensorBoard:

```bash
bash scripts/run_tensorboard_autoresearch.sh
```

## H100 Rehearsals And Submission Packaging

Once you have a strong candidate branch, the next step is to rehearse it against the actual challenge shape:

- `1xH100` for transfer confirmation
- `8xH100` for the real `10 minute` training/eval regime
- a packaged `records/...` candidate folder you can turn into a PR

### 1. Train On 1xH100 For 10 Minutes

```bash
RUN_ID=h100_1x_trial bash scripts/run_h100_1x_train.sh
```

This writes:

- `./runs/runpod_h100_1x_10min/<run_id>/results.json`
- `./runs/runpod_h100_1x_10min/<run_id>/train.log`
- `./runs/runpod_h100_1x_10min/<run_id>/submission_bundle/`

### 2. Train On 8xH100 For 10 Minutes

```bash
RUN_ID=h100_8x_trial bash scripts/run_h100_8x_train.sh
```

This uses `torchrun --standalone --nproc_per_node=8` under the hood.

By default, the `8xH100` launcher now runs in submission-style train mode:

- no periodic validation
- no eval-on-step-0
- no LAWA eval inside the timed train phase
- export the artifact, then score it in a separate eval run

If you want the older full-rehearsal behavior with train-time validation, set:

```bash
SUBMISSION_TRAIN_ONLY=0 RUN_ID=h100_8x_trial bash scripts/run_h100_8x_train.sh
```

### 3. Re-Evaluate The Exported Artifact Under The Eval Path

Use a separate eval run so you have an exact artifact-reload result and wall-clock number for evaluation:

```bash
ARTIFACT_PATH=./runs/runpod_h100_8x_10min/h100_8x_trial/submission_bundle \
RUN_ID=h100_8x_eval \
bash scripts/run_h100_8x_eval.sh
```

The eval launcher will automatically use the `config.json` saved beside the training run artifact when available, so eval uses the exact exported model config instead of only the base H100 config.

### 4. Package A Candidate Records Folder

```bash
uv run pgolf-package-submission \
  --train_results_json ./runs/runpod_h100_8x_10min/h100_8x_trial/results.json \
  --eval_results_json ./runs/runpod_h100_8x_eval/h100_8x_eval/results.json \
  --track track_10min_16mb \
  --name "My Candidate" \
  --author "Your Name" \
  --github_id your-github-id \
  --blurb "Short summary of the method."
```

This creates:

- `./submission_candidates/track_10min_16mb/<date>_<slug>/README.md`
- `./submission_candidates/track_10min_16mb/<date>_<slug>/submission.json`
- `./submission_candidates/track_10min_16mb/<date>_<slug>/train_gpt.py`
- copied run logs, configs, manifests, and results JSONs

The package is intended to match the official Parameter Golf PR shape:

- `README.md`
- `submission.json`
- train log(s)
- `train_gpt.py` and any counted dependencies  
Source: [official README](https://github.com/openai/parameter-golf/blob/main/README.md)

Important caveats from the official rules:

- record-track runs must train in under `10 minutes` on `8xH100`
- evaluation has its own separate `10 minute` limit
- artifact size is `code bytes + compressed model bytes`
- all counted code should live in `train_gpt.py`
- no external downloads, dataset access, or network calls are allowed during evaluation  
Source: [official README](https://github.com/openai/parameter-golf/blob/main/README.md)

This serves the autoresearch run tree on port `6006` and lets you compare current and previous runs in one place.

Each run writes TensorBoard events under:

- `./runs/autoresearch_5090/runs/<run_id>/tensorboard`
- `./runs/runpod_5090_single_gpu/tensorboard`

## Why This Repo Exists

The official `openai/parameter-golf` repo is the source of truth for:

- challenge rules
- allowed training/eval budgets
- official cached data layout
- submission requirements

This repo exists as an ongoing experiment setup:

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
- `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md`: one-iteration prompt for the overnight loop
- `AUTORESEARCH_SETUP.md`: longer setup notes
- `scripts/autoresearch_state.py`: lightweight Ralph-style session state helper
- `scripts/init_autoresearch_session.sh`: creates `.autoresearch/session.json`
- `scripts/run_codex_autoresearch_loop.sh`: fresh `codex exec` supervisor loop
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
make init-autoresearch-session
make tensorboard-autoresearch
make compare-autoresearch
```

## Notes

- This repo is an ongoing experiment repo, not a finished leaderboard submission.
- The default counted code path is just `train.py`, which keeps artifact accounting close to Parameter Golf expectations.
- The 5090 loop is a production-proxy search tier, not the final target itself.
- The final contest objective is still the official one from `openai/parameter-golf`.
