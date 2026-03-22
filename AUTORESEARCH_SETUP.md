# Autoresearch Setup

## Intended Shape

This repo is structured so an external search loop can treat:

- `train.py` as the main editable target
- `prepare.py` as fixed prep/eval utilities
- `program.md` as the behavioral contract
- `results.json` and `crash.json` as the control surface

For Codex specifically, use:

- `CODEX_AUTORESEARCH_PROMPT.md` as the copy/paste bootstrap prompt
- `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md` for the non-interactive loop runner
- `program.md` as the durable search policy / contract
- `COMPETITIVE_PRIORS.md` as the current leaderboard-aligned research brief

## Recommended Setup

This is the intended Codex workflow on a 5090 host.

1. Create an environment:

```bash
bash scripts/bootstrap.sh
```

2. Download the official challenge data once:

```bash
TRAIN_SHARDS=80 bash scripts/download_official_fineweb.sh
```

3. Run one fixed-time 5090 baseline with the autoresearch wrapper:

```bash
RUN_ID=baseline_5090_5min bash scripts/run_autoresearch_experiment.sh
```

4. Initialize the lightweight Ralph-style session state:

```bash
bash scripts/init_autoresearch_session.sh
```

This creates:

- `./.autoresearch/session.json`
- `./.autoresearch/experiments.jsonl`
- `./.autoresearch/notes.md`
- `./state/autoresearch/accepted_state.json`
- `./configs/promoted/autoresearch_5090_best.json`
- `./configs/promoted/autoresearch_h100_8x_best.json`
- `./configs/promoted/autoresearch_h100_1x_best.json`

The session file now carries a production-aligned search policy, including:

- a soft 5090 artifact target band of `12,000,000` to `15,500,000` bytes
- a `~0.001 val_bpb` meaningful-win threshold
- a limit of `3` consecutive losing micro-tuning runs before the next run should be structural / byte-allocation oriented

Once initialized, non-baseline autoresearch runs will refuse to start unless `session.json` is in `ready` state.

On a fresh 5090 host with no old `runs/` tree, the same init script will fall back to `./state/autoresearch/accepted_state.json` and recreate `./.autoresearch/session.json` from the git-tracked accepted winner.

If you need to refresh those tracked files from the current local accepted winner before pushing, run:

```bash
.venv/bin/python scripts/autoresearch_state.py --state_dir ./.autoresearch sync-tracked-accepted
```

5. Optional live monitor in a second terminal:

```bash
bash scripts/run_tensorboard_autoresearch.sh
```

This serves TensorBoard on port `6006` for the whole autoresearch run tree, so you can compare the current run against previous runs. On a remote host, forward the port to your local machine, for example:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<host>
```

Then open the forwarded TensorBoard URL locally.

6. Start Codex in the repo root.

Recommended command on a dedicated remote box you control:

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

Then paste the prompt from `CODEX_AUTORESEARCH_PROMPT.md`.

Codex should create and manage its own dedicated autoresearch branch for the session.

For overnight/autonomous execution, prefer the Ralph-style supervisor loop instead of one long interactive Codex chat:

```bash
bash scripts/run_codex_autoresearch_loop.sh
```

This launches a fresh `codex exec` for each iteration using `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md`.

The loop writes additional local state under:

- `./.autoresearch/activity.log`
- `./.autoresearch/errors.log`
- `./.autoresearch/runs/`

For a read-only live tail of the supervisor plus the newest iteration log:

```bash
bash scripts/watch_codex_autoresearch.sh
```

7. Read results from:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`
- the concrete run directory referenced by those index files
- `./runs/autoresearch_5090/results.tsv`
- `./runs/autoresearch_5090/runs/<run_id>/metrics.jsonl`
- `./.autoresearch/session.json`
- `./.autoresearch/experiments.jsonl`
- `./.autoresearch/notes.md`

6. Promote only clear 5090 winners to H100 and then 8xH100 rehearsal runs.

The default git-tracked promotion files are:

- `./configs/promoted/autoresearch_5090_best.json`
- `./configs/promoted/autoresearch_h100_8x_best.json`
- `./configs/promoted/autoresearch_h100_1x_best.json`

## Agent Guidance

- Prefer editing `train.py` only.
- Read `program.md` before mutating anything.
- Read `CODEX_AUTORESEARCH_PROMPT.md` when starting a fresh Codex session.
- Read `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md` when running the supervised overnight loop.
- Reject stdout scraping. Use structured files.
- Keep experiments short and comparable under a fixed wall-clock budget.
- Use `scripts/run_autoresearch_experiment.sh` instead of directly invoking long manual runs.
- Treat the 5090 loop as a production-proxy search tier, not a toy sandbox.
- Use git history as experiment memory.
- Commit every experiment attempt before running it.
- Keep winners as normal commits and revert losers with normal revert commits so the full search history stays visible.
- Use `.autoresearch/session.json` as the readiness gate before starting new experiments.
- Use `.autoresearch/experiments.jsonl` as append-only local experiment telemetry.
- Use `.autoresearch/notes.md` as the structural hypothesis ledger for open, tried, winning, and rejected ideas.
- After an accepted winner, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`.
- In a fresh session, cover structural axes early: `d_model`, `shared_layers` vs `recurrence_loops`, `tail_layers`, `mlp_mult`, `adapter_rank` / `adapter_targets`, and fake-quant timing / clip percentile.
- If artifact size remains below the search target band, prefer bounded architecture / byte-allocation experiments before long stretches of optimizer micro-tuning.
- If resuming, point `--resume_from` at `output_dir/checkpoints/final.pt` or `latest.pt`.

## Useful Commands

Validate a results file:

```bash
.venv/bin/python validate_results.py ./runs/autoresearch_5090/index/latest.json
```

Evaluate a saved artifact:

```bash
.venv/bin/python prepare.py eval-artifact \
  --config_json ./configs/autoresearch_5090_5min.json \
  --artifact_path ./runs/autoresearch_5090/index/best_run/submission_bundle \
  --output_dir ./runs/autoresearch_5090_eval
```

Summarize submission bytes:

```bash
.venv/bin/python summarize_artifact.py --results_json ./runs/autoresearch_5090/index/best.json
```

Compare completed runs:

```bash
.venv/bin/python compare_runs.py --results_tsv ./runs/autoresearch_5090/results.tsv
```

## H100 Rehearsal Path

Once a candidate survives the cheap search loop, use the dedicated H100 launchers:

```bash
CONFIG_JSON=./configs/promoted/autoresearch_h100_1x_best.json RUN_ID=h100_1x_trial bash scripts/run_h100_1x_train.sh
CONFIG_JSON=./configs/promoted/autoresearch_h100_8x_best.json RUN_ID=h100_8x_trial bash scripts/run_h100_8x_train.sh
ARTIFACT_PATH=./runs/runpod_h100_8x_10min/h100_8x_trial/submission_bundle RUN_ID=h100_8x_eval bash scripts/run_h100_8x_eval.sh
```

The `8xH100` train launcher defaults to submission-style timing:

- no train-time periodic validation
- no eval-first-step
- no LAWA eval during the train phase

This keeps the timed train run closer to the official challenge shape, where scoring happens in a separate eval pass.

Then package a records-folder candidate:

```bash
.venv/bin/python package_submission_candidate.py \
  --train_results_json ./runs/runpod_h100_8x_10min/h100_8x_trial/results.json \
  --eval_results_json ./runs/runpod_h100_8x_eval/h100_8x_eval/results.json \
  --track track_10min_16mb \
  --name "My Candidate" \
  --author "Your Name" \
  --github_id your-github-id \
  --blurb "Short summary of the method."
```

This emits `./submission_candidates/...` with `README.md`, `submission.json`, `train_gpt.py`, and copied logs/results/manifests.
