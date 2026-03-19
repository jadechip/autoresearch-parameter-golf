# External Agent Instructions

## What You May Edit

- Edit `train.py`.
- You may change hyperparameters, defaults, schedules, adapter placement, quantization behavior, and model/training details inside `train.py`.
- Keep the default architecture recognizable unless a replacement is clearly better and backed by measured results.

## What You Must Not Rely On

- Do not scrape raw stdout as the control surface.
- Do not depend on `run.log` tailing for success/failure detection.
- Do not assume crashes will only be visible in terminal output.

Use structured files instead:

- `output_dir/results.json`
- `output_dir/crash.json`
- `results.tsv`
- `output_dir/submission_bundle/manifest.json`
- `runs/autoresearch_5090/index/latest.json`
- `runs/autoresearch_5090/index/best.json`

## Files That Should Usually Stay Fixed

- `prepare.py`
- `validate_results.py`
- `summarize_artifact.py`
- `scripts/run_autoresearch_experiment.sh`
- `scripts/index_autoresearch_run.py`
- `program.md`
- `README.md`
- `AUTORESEARCH_SETUP.md`
- `PRODUCTION_READINESS.md`
- tests, unless a requested code change requires matching test updates

## Non-Negotiable Invariants

- Preserve tokenizer-agnostic bits-per-byte correctness.
- Keep `results.json` schema valid.
- Keep export -> reload -> eval working.
- Keep checkpoint save/resume working.
- Crash fast on invalid configs.
- Keep `train.py` as the main mutation target.

## Conservative Search Space

Prioritize these knobs first:

- `d_model`
- `shared_layers`
- `recurrence_loops`
- `adapter_rank`
- `adapter_alpha`
- `fake_quant_start_step`
- optimizer LR values
- `train_batch_tokens`
- `grad_accum_steps`
- `quant.clip_percentile`

## Search Policy

- Use the 5090 autoresearch loop for broad search:
  `bash scripts/run_autoresearch_experiment.sh`
- Keep experiments on a fixed 300-second budget.
- Use `runs/autoresearch_5090/index/latest.json` and `best.json` as the agent-readable state.
- Prefer simpler wins first.
- Reject hacks that add complexity for tiny gains.
- Keep code changes local and legible.
- If a change risks export/reload parity or structured outputs, do not ship it without tests.
- If a run fails, inspect `crash.json` and `results.json` before changing code.
- Promote only meaningful wins from 5090 search to H100 validation, then to 8xH100 submission rehearsals.

## Parameter Golf Alignment

- Default artifact accounting counts `train.py` plus the compressed model blob.
- Submission-oriented bundle output lives in `output_dir/submission_bundle/`.
- Keep byte accounting honest: count exact shipped bytes, not rough estimates.
- Treat 5090 autoresearch as a development tier, not a record-valid run tier.
- Final record attempts must be reproduced under the official 8xH100 time limits.
