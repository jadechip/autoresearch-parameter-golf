# Codex Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for a single Ralph-style autoresearch iteration.

The outer supervisor should launch a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, and COMPETITIVE_PRIORS.md first.

You are running exactly one autoresearch iteration in this repo. Do not loop forever in this invocation.

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated autoresearch branch, create one yourself with a timestamped name like autoresearch/20260320-010000
- inspect recent commits on that branch
- run `.venv/bin/python scripts/summarize_recent_autoresearch.py --limit 12` and treat its warnings as hard signals about recent over-exploration of one family
- inspect `.autoresearch/session.json` and do not start unless it exists and `status=ready`
- inspect `.autoresearch/session.json["search_policy"]`
- inspect `.autoresearch/session.json["search_policy"]["campaign_stories"]`
- inspect `.autoresearch/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090/index/best.json`

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision, stop

Git policy:
- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Make one commit for the experiment before running it.
- If the run loses or crashes, preserve it in history and return to accepted state with a normal revert commit.
- Do not use `git reset` to erase experiment history.

Allowed files:
- edit `train.py`
- update `.autoresearch/notes.md`
- a chosen campaign story may add a missing self-contained module inside `train.py`; do not retreat to a safer micro-tune just because the module does not exist yet

Do not modify:
- `prepare.py`
- helper scripts, unless a new architecture capability cannot be tested honestly without a minimal structured-output-safe change
- docs
- dependencies

Primary goal:
- minimize `val_bpb`

Constraints:
- preserve valid export and reloadability
- keep `artifact_bytes < 16000000`
- prefer changes that plausibly transfer to `1xH100` / `8xH100`
- prefer simplicity

Search policy:
- treat `runs/autoresearch_5090/index/best.json` as numeric telemetry, not the sole truth of accepted state
- treat `COMPETITIVE_PRIORS.md` as the current research brief; the recovered compact baseline is now in the wrong regime for a likely win, so bias toward a stronger carrier instead of more shrinkage
- if accepted artifact size is below the target band in `.autoresearch/session.json`, prefer bounded architecture / byte-allocation experiments over endless optimizer micro-tuning
- in a fresh session, cover structural axes early: `d_model`, `shared_layers` vs `recurrence_loops`, `tail_layers`, `mlp_mult`, `adapter_rank` / `adapter_targets`, fake-quant timing / clip percentile
- do not spend more than the search-policy limit of consecutive losing micro-tuning experiments without making the next run structural or byte-allocation oriented
- treat these as micro-tuning, not structural exploration:
  - selective float / fake-quant toggles on one or two tensors of an otherwise unchanged carrier
  - narrowing or widening an existing float/QAT exemption set
  - one-step batch tweaks without a carrier change
- if the recent summary shows one dominant family or a same-family streak of 3+, force the next run into a different branch family
- if the accepted artifact is already above roughly `14 MB`, stop slicing late-tensor float groups finer unless the next run also changes carrier, batch/context, low-rank-Q placement, or local-token structure
- treat `.autoresearch/session.json["search_policy"]["campaign_stories"]` as a Ralph-style story board and pick exactly one story to advance in this invocation
- a campaign story may be module-writing:
  - smarter local-token module
  - XSA or top-layer cache
  - SmearGate or TTT branch
  - Canon or neighbor mixer
- if you pick a module-writing story, prefer a small self-contained implementation in `train.py` over a placeholder commit
- use `.autoresearch/notes.md` as the durable hypothesis ledger
- split the campaign into clean branches instead of one blended soup:
  - near-full-budget carrier
  - late selective quantization / post-quant soup
  - low-rank Q
  - smarter local-token module
- prioritize immediate directions that fit the current repo or can be made to fit with a self-contained module in `train.py`:
  - low-rank Q
  - late selective coarse-group quantization
  - selective higher precision for embeddings / head
  - compute-aware batch / context curricula
  - simpler local-token modules
- actively seed at least some runs into harder branches instead of leaving them as permanent stretch ideas:
  - XSA
  - cross-window or top-layer KV cache
  - SmearGate + TTT-style branches
  - Canon inserts
- deprioritize strategically low-value directions:
  - more recurrence/shared-core looping as a main direction
  - tiny-model compression tricks
  - blunt `d_model` widening
  - full `num_kv_heads=8`
  - near-neighbor context increases above `768` without compute reclamation
  - compensated global `mlp_mult=3`
  - pure tail-width nudges

Required loop for this invocation:
1. inspect branch tip, `.autoresearch/session.json`, `.autoresearch/notes.md`, `COMPETITIVE_PRIORS.md`, recent commits, `best.json`, the recent-family summary, and the campaign stories
2. choose one concrete next hypothesis by selecting one campaign story
3. edit `train.py`
4. commit the experiment
5. run exactly one `bash scripts/run_autoresearch_experiment.sh`
6. inspect `latest.json`, `best.json`, and the concrete run outputs
7. keep the experiment commit if it wins, otherwise add a revert commit
8. record the decision with:
   `.venv/bin/python scripts/autoresearch_state.py --state_dir ./.autoresearch decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
9. if the run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`
10. update `.autoresearch/notes.md` with a short note about the tested hypothesis and outcome
11. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout or run.log as the control surface.
Use structured files and git history as memory.
```
