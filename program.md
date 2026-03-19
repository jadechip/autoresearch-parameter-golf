# External Agent Instructions

This repo uses a two-tier workflow:

- `1x5090` autoresearch is a production-proxy search tier.
- `1xH100` and `8xH100` runs are promotion / submission-rehearsal tiers.

Your job on the 5090 tier is not to win the final challenge directly. Your job is to find changes that are likely to transfer to the real Parameter Golf constraints:

- train under 10 minutes on `8xH100 SXM`
- eval under 10 minutes
- artifact under `16,000,000` bytes
- lowest possible `val_bpb`

At the start of a session, inspect:

- current git branch
- recent git history on the autoresearch branch
- `runs/autoresearch_5090/index/best.json`

Use git history as the experiment memory for what has already been tried and what has already won.

## What You May Edit

- Edit `train.py`.
- You may change hyperparameters, defaults, schedules, adapter placement, quantization behavior, and model/training details inside `train.py`.
- Architecture changes inside the current baseline family are allowed and expected.
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

Keep the baseline family intact:

- decoder-only LM
- tied embeddings
- GQA
- RoPE
- RMSNorm
- ReLU^2 MLP
- depth recurrence via shared blocks
- per-loop adapters
- QAT-aware training
- row-wise INT8 export + zlib packing

You may reallocate capacity within this family, but do not wander into unrelated architecture rewrites.

## Production-Proxy Objective

Primary objective:

- lower `val_bpb`

Subject to:

- valid export
- valid reload / evaluation path
- `artifact_bytes < 16,000,000`
- no unreasonable VRAM explosion
- simplicity preference

This repo's current 5090 baseline materially under-spends the final artifact budget. Do not assume the best search strategy is to keep the model tiny and only polish optimizer settings. Favor ideas that plausibly improve the final `8xH100` submission regime, including sensible use of additional bytes.

## Search Space

Prioritize production-relevant knobs first:

- `d_model`
- `shared_layers`
- `recurrence_loops`
- `tail_layers`
- `mlp_mult`
- `adapter_rank`
- `adapter_alpha`
- `adapter_targets`
- `fake_quant_start_step`
- quantization `clip_percentile`
- optimizer LR values
- `train_batch_tokens`
- `grad_accum_steps`

If artifact usage remains far below the cap, prefer bounded architecture / byte-allocation experiments before endless micro-tuning of optimizer settings.

Avoid spending time on:

- 5090-specific hacks that are unlikely to transfer to H100
- changes that mainly exploit logging / harness quirks
- high-complexity tweaks for tiny gains
- changes that make final single-script submission meaningfully uglier

## Search Policy

- Work on a dedicated autoresearch branch.
- Before proposing a new mutation, inspect recent commits so you do not waste runs repeating the same idea.
- Use the 5090 autoresearch loop for broad search:
  `bash scripts/run_autoresearch_experiment.sh`
- Keep experiments on a fixed 300-second budget.
- Use `runs/autoresearch_5090/index/latest.json` and `best.json` as the agent-readable state.
- Prefer simpler wins first.
- Treat improvements smaller than roughly `0.001 val_bpb` as noise unless they also simplify the code or clearly move the model toward a better production regime.
- Reject hacks that add complexity for tiny gains.
- Keep code changes local and legible.
- If a change risks export/reload parity or structured outputs, do not ship it without tests.
- If a run fails, inspect `crash.json` and `results.json` before changing code.
- If a run is worse or not meaningfully better, revert the `train.py` change and move on.
- Keep the working state near the current best candidate, not a chain of accumulated regressions.
- Promote only meaningful wins from 5090 search to H100 validation, then to 8xH100 submission rehearsals.

Git discipline:

- If a run is a meaningful winner, commit the `train.py` change to the autoresearch branch.
- Include the run id and resulting `val_bpb` in the commit message.
- If a run loses or fails, return the working tree to the previous good state before trying the next idea.
- Do not accumulate multiple speculative edits without a run in between.
- Treat `best.json` plus recent commits as the authoritative memory of current best state.

Among candidates with similar `val_bpb`, prefer:

1. simpler code
2. more production-plausible architecture choices
3. better use of the available artifact budget
4. lower VRAM

## Promotion Guidance

Use the 5090 tier to identify candidates worth replaying on H100:

- clear `val_bpb` win
- no export/reload regressions
- no artifact accounting issues
- no obvious dependence on 5090-only behavior

Do not treat the 5090 tier as the final contest metric. It is a filter for what deserves expensive H100 time.

## Parameter Golf Alignment

- Default artifact accounting counts `train.py` plus the compressed model blob.
- Submission-oriented bundle output lives in `output_dir/submission_bundle/`.
- Keep byte accounting honest: count exact shipped bytes, not rough estimates.
- Treat 5090 autoresearch as a development tier, not a record-valid run tier.
- Final record attempts must be reproduced under the official 8xH100 time limits.
