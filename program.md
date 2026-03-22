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
- if needed, create a dedicated autoresearch branch for the session
- recent git history on that autoresearch branch
- `.autoresearch/session.json`
- `.autoresearch/notes.md`
- `COMPETITIVE_PRIORS.md`
- `runs/autoresearch_5090/index/best.json`

Use git history as the experiment memory for what has already been tried and what has already won.
Use `.venv/bin/python scripts/summarize_recent_autoresearch.py --limit 12` as a compact novelty check before choosing the next mutation.

Accepted-state policy:

- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Treat `runs/autoresearch_5090/index/best.json` as numeric telemetry only. It may reflect a reverted run.
- Do not let `best.json` override the accepted branch state on its own.
- Treat `.autoresearch/session.json` as the readiness gate. If it is missing or not `ready`, do not start the loop.

## What You May Edit

- Edit `train.py`.
- You may also update `.autoresearch/notes.md` as local session memory for hypotheses and outcomes.
- You may change hyperparameters, defaults, schedules, adapter placement, quantization behavior, and model/training details inside `train.py`.
- Architecture changes inside the current baseline family are allowed and expected.
- Self-contained module additions inside `train.py` are also allowed when they preserve autoregressive causality, export/reload, artifact accounting, and structured results.
- Keep the decoder-only submission regime recognizable unless a replacement is clearly better and backed by measured results.

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
- `.autoresearch/session.json`
- `.autoresearch/experiments.jsonl`
- `.autoresearch/notes.md`

## Files That Should Usually Stay Fixed

- `prepare.py`
- `validate_results.py`
- `summarize_artifact.py`
- `scripts/autoresearch_state.py`
- `scripts/init_autoresearch_session.sh`
- `scripts/run_codex_autoresearch_loop.sh`
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

Keep the competition invariants intact:

- decoder-only LM
- tied embeddings
- GQA
- RoPE
- RMSNorm
- ReLU^2 MLP
- depth recurrence or shallow shared-core variants via shared blocks
- per-loop adapters
- QAT-aware training
- row-wise INT8 export + zlib packing

You may reallocate capacity within this family and add self-contained local modules that still fit the same submission and export discipline. Do not wander into unrelated training stacks or infra rewrites.

Current recovered search baseline:

- the last trusted compact 5090 winner was a `seq_len=768`, `1 shared x 1 loop`, `tail=3`, `8x` unique-tail MLP line at about `7.26 MB`
- that materially under-spends the hard `16 MB` cap
- do not assume the best next move is to keep shrinking the model
- do not assume more recurrence is the right frontier direction either

## Production-Proxy Objective

Primary objective:

- lower `val_bpb`

Subject to:

- valid export
- valid reload / evaluation path
- `artifact_bytes < 16,000,000`
- no unreasonable VRAM explosion
- simplicity preference

This repo's current compact 5090 baseline materially under-spends the final artifact budget. Do not assume the best search strategy is to keep the model tiny and only polish optimizer settings. Favor ideas that plausibly improve the final `8xH100` submission regime, including deliberate use of additional bytes.

Use the search policy recorded in `.autoresearch/session.json`:

- soft artifact target band for 5090 search: `12,000,000` to `15,500,000` bytes
- meaningful improvement threshold: about `0.001 val_bpb`
- maximum consecutive losing micro-tuning runs before a required structural follow-up: `3`

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

Prioritize leaderboard-aligned directions that still fit inside `train.py`:

- low-rank Q as a structural reallocation tool
- mixed low-bit quantization beyond MLP-only export
- selective higher-precision embeddings / head
- selective or compensated `~3x` MLP families rather than blanket global `mlp_mult=3`
- longer context or sliding-window eval when compute is reclaimed elsewhere
- compute-aware batch / context curricula
- stronger optimizer bundles with Muon momentum, weight decay, warmdown, and possibly SWA after a structural candidate exists
- frontier-style initialization or gating ideas that are local to `train.py`
- simpler local-token modules with better byte/quality tradeoffs than raw scaling of the current line

Split the next serious search into clean branches instead of one blended stack:

- near-full-budget carrier
- late selective quantization / post-quant soup
- low-rank Q
- smarter local-token module
- XSA or top-layer cache
- SmearGate / TTT
- Canon or neighboring-token mixer

Treat the branch list in `.autoresearch/session.json["search_policy"]["campaign_stories"]` as a Ralph-style story board:

- pick exactly one story per iteration
- do not blur multiple module-writing stories into one change
- if recent history is narrow, force a story switch rather than another local refinement
- if the chosen story needs a missing self-contained module in `train.py`, implement it instead of collapsing back to a safe micro-tune

In a fresh session, the first search block should deliberately cover structural axes before settling into local hill-climbing:

- `d_model`
- `shared_layers` vs `recurrence_loops`
- `tail_layers`
- `mlp_mult`
- `adapter_rank` / `adapter_targets`
- `fake_quant_start_step` / `clip_percentile`

Avoid spending time on:

- 5090-specific hacks that are unlikely to transfer to H100
- changes that mainly exploit logging / harness quirks
- high-complexity tweaks for tiny gains
- changes that make final single-script submission meaningfully uglier
- repeating known-losing or strategically low-value moves without a new major hypothesis:
  - more recurrence/shared-core looping as a main direction
  - tiny-model compression tricks
  - blunt `d_model` widening
  - full `num_kv_heads=8`
  - near-neighbor context increases above `768` without compute reclamation
  - compensated global `mlp_mult=3`
  - pure tail-width nudges

## Search Policy

- Work on a dedicated autoresearch branch. If one does not already exist for the session, create it yourself.
- Before proposing a new mutation, inspect recent commits so you do not waste runs repeating the same idea.
- Before proposing a new mutation, inspect `.autoresearch/notes.md` so you do not waste runs repeating the same weak structural idea.
- Use the 5090 autoresearch loop for broad search:
  `bash scripts/run_autoresearch_experiment.sh`
- Keep experiments on a fixed 300-second budget.
- Use `runs/autoresearch_5090/index/latest.json` and `best.json` as the agent-readable state.
- Prefer simpler wins first.
- Treat improvements smaller than roughly `0.001 val_bpb` as noise unless they also simplify the code or clearly move the model toward a better production regime.
- Reject hacks that add complexity for tiny gains.
- If accepted artifact size is still below the policy target band, do not spend long stretches only micro-tuning optimizer values.
- Do not spend more than `3` consecutive losing micro-tuning runs without making the next run a structural / byte-allocation experiment.
- Treat selective float / fake-quant toggles on one or two tensors of the same carrier as micro-tuning, not structural exploration.
- If the recent-family summary shows a dominant family or a same-family streak of `3+`, force the next run into a different branch family.
- If the branch switch points to a module-writing story, prefer a minimal working implementation in `train.py` over another carrier-local precision retune.
- If the accepted artifact is already above roughly `14 MB`, stop slicing late-tensor float groups finer unless the same run also changes carrier structure, batch/context, low-rank-Q placement, or local-token structure.
- Alternate broader structural probes with local refinements once you find a promising larger-capacity direction.
- Keep code changes local and legible.
- If a change risks export/reload parity or structured outputs, do not ship it without tests.
- If a run fails, inspect `crash.json` and `results.json` before changing code.
- If a run is worse or not meaningfully better, revert the `train.py` change and move on.
- Keep the working state near the current best candidate, not a chain of accumulated regressions.
- Promote only meaningful wins from 5090 search to H100 validation, then to 8xH100 submission rehearsals.

Git discipline:

- Before each run, commit the exact `train.py` experiment to the autoresearch branch.
- If a run is a meaningful winner, keep that experiment commit as part of the accepted branch history.
- If a run loses or fails, preserve the failed attempt in history and return to the accepted state with a normal revert commit.
- Include the run id and resulting `val_bpb` in keep/revert commit messages when practical.
- Do not use `git reset` to erase experiment history during the loop.
- Do not accumulate multiple speculative edits without a run in between.
- Treat the current branch tip plus recent commits as the authoritative memory of accepted state.
- After each keep/revert decision, update `.autoresearch/session.json` with:
  `.venv/bin/python scripts/autoresearch_state.py --state_dir ./.autoresearch decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
- If a run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/` so the next 5090 or H100 host can recover the winner from git alone.
- After each keep/revert decision, update `.autoresearch/notes.md` with a short hypothesis note so the next fresh Codex iteration can see which structural directions are still open.

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
