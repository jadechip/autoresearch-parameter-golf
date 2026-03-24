# Codex Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for a single Ralph-style autoresearch iteration.

The outer supervisor should launch a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, COMPETITIVE_PRIORS.md, and `.autoresearch/session.json` first.

You are running exactly one autoresearch iteration in this repo. Do not loop forever in this invocation.

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated autoresearch branch, create one yourself with a timestamped name like autoresearch/20260320-010000
- inspect recent commits on that branch
- run `.venv/bin/python scripts/summarize_recent_autoresearch.py --limit 12` and treat its warnings as hard signals about recent over-exploration of one family
- inspect `.autoresearch/session.json` and do not start unless it exists and `status=ready`
- inspect `.autoresearch/session.json["search_policy"]`, including `lane`, `preflight`, `discovery_mode`, `refinement_gate`, and `campaign_stories`, and `train_config_preferences`
- inspect `.autoresearch/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090/index/latest.json` and `best.json`

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision, stop

Git policy:
- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Make one commit for the experiment before running it.
- If the run loses, fails, or is rejected at preflight, preserve it in history and return to accepted state with a normal revert commit.
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

Controller model:
- the aggressive loop is now discovery-first; treat it as a feeder for branch establishment
- the ordinary loop is allowed to refine real contenders, but only after a branch is ranking-valid and plausibly near the accepted line
- use `search_policy["refinement_gate"]` literally: only refine when a contender is real, otherwise keep doing structural branch work
- treat public frontier patterns as priors, not recipes
- set `config_transform_profile` explicitly for new branches; prefer `manual` or `safe_rebalance`, and use `legacy_lineage` only when intentionally revisiting the old compact family
- keep at least one meaningful difference from public winning solutions in carrier shape, scheduling, placement, quant grouping, or local mechanism

Preflight model:
- `run_codex_autoresearch_experiment.sh` may run a benchmark preflight before the full proxy
- if the script exits after a benchmark-only run, inspect `runs/autoresearch_5090/index/latest.json` and the run's `preflight` payload or `preflight_decision.json`
- a benchmark-mode result with `preflight.decision=skip` counts as the one completed experiment for this invocation
- benchmark preflight rejection means the branch was too slow or too over-cap for the current proxy policy; revert it unless the preflight evidence suggests one obvious shrink or simplification for next time

Search policy:
- treat `runs/autoresearch_5090/index/best.json` as numeric telemetry, not the sole truth of accepted state
- treat `.autoresearch/session.json["search_policy"]` as the live campaign brief
- if accepted artifact size is below the target band, prefer bounded architecture / byte-allocation experiments over endless optimizer micro-tuning
- if recent history is narrow, force a different branch family before another local refinement
- if a contender branch already exists, prefer local post-quant / checkpoint-selection / carrier-local refinement over launching another huge redesign in this loop
- otherwise pick exactly one campaign story and advance it with one honest structural attempt
- split the campaign into clean branches instead of one blended soup:
  - near-full-budget carrier
  - low-rank Q reallocation
  - late selective quantization / post-quant selection
  - batch / context curriculum
  - smarter local-token module
  - activation / rope / residual-scaling variants that stay local to `train.py`
  - packed mixed-bit export and per-pattern clip overrides that stay export-clean
  - XSA or top-layer cache
  - SmearGate / TTT
  - Canon or neighboring-token mixer
- deprioritize strategically low-value directions:
  - more recurrence/shared-core looping as a main direction
  - tiny-model compression tricks
  - blunt `d_model` widening without a branch story
  - pure tail-width nudges on the same carrier

Required loop for this invocation:
1. inspect branch tip, `.autoresearch/session.json`, `.autoresearch/notes.md`, `COMPETITIVE_PRIORS.md`, recent commits, `best.json`, the recent-family summary, and the campaign stories
2. decide whether this iteration is branch establishment or contender refinement
3. choose one concrete next hypothesis by selecting one campaign story or one contender-local refinement story
4. edit `train.py`
5. commit the experiment
6. run exactly one `bash scripts/run_autoresearch_experiment.sh`
7. inspect `latest.json`, `best.json`, and the concrete run outputs
8. if the run is a benchmark-only preflight reject, revert and record why it failed the gate
9. if the run is a full training result and wins, keep the experiment commit; otherwise add a revert commit
10. record the decision with:
   `.venv/bin/python scripts/autoresearch_state.py --state_dir ./.autoresearch decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
11. if the run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`
12. update `.autoresearch/notes.md` with a short note about the tested hypothesis and outcome
13. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout or run.log as the control surface.
Use structured files and git history as memory.
```
