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

The session file now carries a production-aligned search policy, including:

- a soft 5090 artifact target band of `7,000,000` to `12,000,000` bytes
- a `~0.001 val_bpb` meaningful-win threshold
- a limit of `3` consecutive losing micro-tuning runs before the next run should be structural / byte-allocation oriented

Once initialized, non-baseline autoresearch runs will refuse to start unless `session.json` is in `ready` state.

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
- In a fresh session, cover structural axes early: `d_model`, `shared_layers` vs `recurrence_loops`, `tail_layers`, `mlp_mult`, `adapter_rank` / `adapter_targets`, and fake-quant timing / clip percentile.
- If artifact size remains below the search target band, prefer bounded architecture / byte-allocation experiments before long stretches of optimizer micro-tuning.
- If resuming, point `--resume_from` at `output_dir/checkpoints/final.pt` or `latest.pt`.

## Useful Commands

Validate a results file:

```bash
uv run pgolf-validate-results ./runs/autoresearch_5090/index/latest.json
```

Evaluate a saved artifact:

```bash
uv run pgolf-prepare eval-artifact \
  --config_json ./configs/autoresearch_5090_5min.json \
  --artifact_path ./runs/autoresearch_5090/index/best_run/submission_bundle \
  --output_dir ./runs/autoresearch_5090_eval
```

Summarize submission bytes:

```bash
uv run pgolf-summarize-artifact --results_json ./runs/autoresearch_5090/index/best.json
```

Compare completed runs:

```bash
uv run pgolf-compare-runs --results_tsv ./runs/autoresearch_5090/results.tsv
```
