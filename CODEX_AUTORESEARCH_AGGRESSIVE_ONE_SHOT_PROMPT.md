# Codex Aggressive Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for one aggressive idea-focused autoresearch iteration.

The outer supervisor launches a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, COMPETITIVE_PRIORS.md, AGGRESSIVE_AUTORESEARCH_IDEAS.md, and `$STATE_DIR/session.json` first.

You are running exactly one aggressive autoresearch iteration in this repo. Do not loop forever in this invocation.

Environment assumptions:
- the outer launcher exported `STATE_DIR` and `AUTORESEARCH_ROOT`
- these point to the aggressive campaign state and aggressive run root
- use them instead of assuming the default `.autoresearch` paths

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated aggressive autoresearch branch, create one yourself with a timestamped name like aggressive-autoresearch/20260323-010000
- inspect recent commits on that branch
- inspect `$STATE_DIR/session.json` and do not start unless it exists and `status=ready`
- inspect `$STATE_DIR/session.json["search_policy"]`, especially `lane`, `preflight`, `creative_guardrails`, and `refinement_gate`, and `train_config_preferences`
- inspect `$STATE_DIR/aggressive_campaign.json`
- identify the current active idea from `$STATE_DIR/aggressive_campaign.json`
- inspect `configs/aggressive_autoresearch_ideas.json`
- inspect `AGGRESSIVE_AUTORESEARCH_IDEAS.md`
- inspect `$STATE_DIR/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090_aggressive/index/latest.json` and `best.json` if they exist
- inspect the active idea via:
  `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" current`
- read its `next_attempt_blueprint` and treat that blueprint as mandatory

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision and campaign bookkeeping, stop

Discovery-first policy:
- this loop is for discovering real new basins, not for producing the final winner directly
- use it to establish viable branches and to generate ranking-valid contenders
- once a branch is ranking-valid and clearly contender-like, do not spend this loop polishing it forever; record it and let the ordinary refine loop exploit it
- `recommended_phase=refine` means the idea has already produced a plausible contender; only spend a remaining attempt on refinement if the blueprint explicitly calls for it and the branch still needs one obvious clean test

Git policy:
- Treat the current branch tip as the accepted code state for this aggressive campaign.
- Treat recent git history as the canonical memory of accepted and rejected aggressive experiments.
- Make one commit for the experiment before running it.
- If the run loses, crashes, or is rejected at preflight, preserve it in history and return to accepted state with a normal revert commit.
- Do not use `git reset` to erase experiment history.

Allowed files:
- edit `train.py`
- update `$STATE_DIR/notes.md`
- if the current idea cannot be tested honestly without it, make one minimal helper-script change that preserves structured outputs and export/reload invariants

Do not modify:
- `prepare.py`
- dependencies
- unrelated docs

Aggressive-campaign rules:
- work on exactly one idea: the current active idea in `$STATE_DIR/aggressive_campaign.json`
- do not switch ideas manually
- each idea gets exactly six attempts across loop iterations unless the controller early-stops it
- treat the six attempts as six materially different architecture variants, not a local refinement ladder
- implement the exact `next_attempt_blueprint`; do not pick a safer nearby mutation inside the same story
- if the idea needs a missing self-contained module in `train.py`, implement it instead of shrinking the idea into another precision nudge
- preserve causal scoring, exportability, reloadability, and honest byte accounting
- do not stack multiple hard ideas into one run unless the blueprint explicitly calls for that combination
- keep public frontier patterns as priors, not recipes; preserve at least one clear creative difference in placement, schedule, carrier partition, or local mechanism

Variant validity rule:
- a valid attempt must either introduce the named mechanism from the blueprint or change at least two macro axes from the accepted aggressive branch
- if you cannot describe the attempt as a new architecture variant in one sentence, it is too small
- if the resulting carrier still looks like the current depth-3 family with a few toggles, it is too small

Preflight rule:
- `run_autoresearch_experiment.sh` may stop at a benchmark-only preflight if the branch is obviously too slow or over-cap
- a benchmark-only `latest.json` with `preflight.decision=skip` counts as the completed experiment for this invocation
- treat that as informative telemetry, record why it failed, revert, and move on

Search policy for this aggressive loop:
- favor meaningful branch establishment over tiny score polishing
- the campaign default regime is roughly near-cap artifact usage, 11-12 layers, d_model 512, about 3x MLP, seq_len 2048, and batch around 524K tokens when feasible
- the active blueprint may deliberately step outside the current accepted branch shape; that is expected
- for XSA, cache, SmearGate, TTT, Canon, or smarter local-token stories, isolate the new idea enough that the result is interpretable
- if an idea naturally implies two families, keep them separate rather than blending them immediately
- new train.py-local directions are fair game when they fit the blueprint spirit, such as partial RoPE, activation-family changes, residual-scale initialization, packed mixed low-bit export groups, and per-pattern clip overrides
- set `config_transform_profile` explicitly for any new branch; prefer `manual` or `safe_rebalance`, and only use `legacy_lineage` when a blueprint intentionally revisits the old compact family

Required loop for this invocation:
1. inspect the current aggressive branch tip, `$STATE_DIR/session.json`, `$STATE_DIR/aggressive_campaign.json`, `configs/aggressive_autoresearch_ideas.json`, `AGGRESSIVE_AUTORESEARCH_IDEAS.md`, `$STATE_DIR/notes.md`, and recent commits
2. read `next_attempt_blueprint`, restate it in your own words, and write down which macro axes it changes
3. edit `train.py`
4. commit the experiment, including the idea id in the commit message
5. run exactly one `bash scripts/run_autoresearch_experiment.sh`
6. inspect `latest.json`, `best.json`, and the concrete run outputs under `runs/autoresearch_5090_aggressive/`
7. if the run is a benchmark-only preflight reject, revert and record why it failed the gate
8. if the run is ranking-valid and wins, keep the experiment commit; otherwise add a revert commit
9. record the decision with:
   `.venv/bin/python scripts/autoresearch_state.py --state_dir "$STATE_DIR" decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
10. record the attempt with:
   `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" record-attempt --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
11. if the run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`
12. update `$STATE_DIR/notes.md` with a short note about the tested hypothesis and outcome
13. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout or run.log as the control surface.
Use structured files and git history as memory.
```
