# Autoresearch Setup

## Intended Shape

This repo is structured so an external search loop can treat:

- `train.py` as the main editable target
- `prepare.py` as fixed prep/eval utilities
- `program.md` as the behavioral contract
- `results.json` and `crash.json` as the control surface

For Codex specifically, use:

- `CODEX_AUTORESEARCH_PROMPT.md` as the copy/paste bootstrap prompt
- `program.md` as the durable search policy / contract

## Recommended Setup

1. Create an environment:

```bash
bash scripts/bootstrap.sh
```

2. Download the official challenge data once:

```bash
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
```

3. Run one fixed-time 5090 baseline with the autoresearch wrapper:

```bash
bash scripts/run_autoresearch_experiment.sh
```

Optional live monitor in a second terminal:

```bash
uv run pgolf-watch-run ./runs/autoresearch_5090/index/latest.json
```

4. Start Codex in the repo root and paste the prompt from `CODEX_AUTORESEARCH_PROMPT.md`.

5. Read results from:

- `./runs/autoresearch_5090/index/latest.json`
- `./runs/autoresearch_5090/index/best.json`
- the concrete run directory referenced by those index files
- `./runs/autoresearch_5090/results.tsv`
- `./runs/autoresearch_5090/runs/<run_id>/metrics.jsonl`

6. Promote only clear 5090 winners to H100 and then 8xH100 rehearsal runs.

## Agent Guidance

- Prefer editing `train.py` only.
- Read `program.md` before mutating anything.
- Read `CODEX_AUTORESEARCH_PROMPT.md` when starting a fresh Codex session.
- Reject stdout scraping. Use structured files.
- Keep experiments short and comparable under a fixed wall-clock budget.
- Use `scripts/run_autoresearch_experiment.sh` instead of directly invoking long manual runs.
- Treat the 5090 loop as a production-proxy search tier, not a toy sandbox.
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
