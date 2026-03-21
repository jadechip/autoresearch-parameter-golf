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
- inspect `.autoresearch/session.json` and do not start unless it exists and `status=ready`
- inspect `.autoresearch/session.json["search_policy"]`
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

Do not modify:
- `prepare.py`
- helper scripts
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
- treat `COMPETITIVE_PRIORS.md` as the current research brief; the recovered compact baseline materially under-spends the hard `16 MB` cap, so bias toward deliberate capacity spending rather than more shrinkage
- if accepted artifact size is below the target band in `.autoresearch/session.json`, prefer bounded architecture / byte-allocation experiments over endless optimizer micro-tuning
- in a fresh session, cover structural axes early: `d_model`, `shared_layers` vs `recurrence_loops`, `tail_layers`, `mlp_mult`, `adapter_rank` / `adapter_targets`, fake-quant timing / clip percentile
- do not spend more than the search-policy limit of consecutive losing micro-tuning experiments without making the next run structural or byte-allocation oriented
- use `.autoresearch/notes.md` as the durable hypothesis ledger
- prioritize leaderboard-aligned directions that still fit `train.py`:
  - mixed low-bit quantization beyond MLP-only export
  - selective wider MLP allocations instead of blanket global `mlp_mult=3`
  - longer context or sliding-window eval if compute is reclaimed
  - selective higher precision for embeddings / head
  - frontier-style init or gating ideas
- deprioritize ideas that already lost on the compact line unless paired with a new major hypothesis:
  - blunt `d_model` widening
  - full `num_kv_heads=8`
  - near-neighbor context increases above `768` without compute reclamation
  - compensated global `mlp_mult=3`
  - pure tail-width nudges

Required loop for this invocation:
1. inspect branch tip, `.autoresearch/session.json`, `.autoresearch/notes.md`, `COMPETITIVE_PRIORS.md`, recent commits, and `best.json`
2. choose one concrete next hypothesis
3. edit `train.py`
4. commit the experiment
5. run exactly one `bash scripts/run_autoresearch_experiment.sh`
6. inspect `latest.json`, `best.json`, and the concrete run outputs
7. keep the experiment commit if it wins, otherwise add a revert commit
8. record the decision with:
   `uv run python scripts/autoresearch_state.py --state_dir ./.autoresearch decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
9. update `.autoresearch/notes.md` with a short note about the tested hypothesis and outcome
10. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout or run.log as the control surface.
Use structured files and git history as memory.
```
