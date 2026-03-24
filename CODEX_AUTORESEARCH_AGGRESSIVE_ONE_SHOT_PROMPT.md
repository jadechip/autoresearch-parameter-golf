# Codex Aggressive Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for one aggressive idea-focused autoresearch iteration.

The outer supervisor launches a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, COMPETITIVE_PRIORS.md, and AGGRESSIVE_AUTORESEARCH_IDEAS.md first.

You are running exactly one aggressive autoresearch iteration in this repo.
Do not loop forever in this invocation.

Environment assumptions:
- the outer launcher exported `STATE_DIR` and `AUTORESEARCH_ROOT`
- these point to the aggressive campaign state and aggressive run root
- use them instead of assuming the default `.autoresearch` paths

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated aggressive autoresearch branch, create one yourself with a timestamped name like aggressive-autoresearch/20260323-010000
- inspect recent commits on that branch
- inspect `$STATE_DIR/session.json` and do not start unless it exists and `status=ready`
- inspect `$STATE_DIR/aggressive_campaign.json`
- identify the current active idea from `$STATE_DIR/aggressive_campaign.json`
- inspect `configs/aggressive_autoresearch_ideas.json`
- inspect `AGGRESSIVE_AUTORESEARCH_IDEAS.md`
- inspect `$STATE_DIR/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090_aggressive/index/best.json` if it exists
- inspect the active idea via:
  `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" current`
- read its `next_attempt_blueprint` and treat that blueprint as mandatory

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision and campaign bookkeeping, stop

Git policy:
- Treat the current branch tip as the accepted code state for this aggressive campaign.
- Treat recent git history as the canonical memory of accepted and rejected aggressive experiments.
- Make one commit for the experiment before running it.
- If the run loses or crashes, preserve it in history and return to accepted state with a normal revert commit.
- Do not use `git reset` to erase experiment history.

Allowed files:
- edit `train.py`
- update `$STATE_DIR/notes.md`
- if the current idea cannot be tested honestly without it, make one minimal helper-script change that preserves structured outputs and export/reload invariants

Do not modify:
- `prepare.py`
- dependencies
- unrelated docs

Primary goal:
- make meaningful progress on the current idea while minimizing `val_bpb`

Aggressive-campaign rules:
- work on exactly one idea: the current active idea in `$STATE_DIR/aggressive_campaign.json`
- do not switch ideas manually
- each idea gets exactly six attempts across loop iterations
- treat the six attempts as six materially different architecture variants, not a local refinement ladder
- the current idea metadata includes:
  - `attempt_mode`
  - `must_change_axes`
  - `forbidden_refinements`
- `current` also exposes:
  - `next_attempt_number`
  - `next_attempt_blueprint`
  - `completed_blueprints`
  - `remaining_blueprints`
  - `ranking_policy`
  - `accepted_baseline`
  - `pareto_frontier`
  - `recommended_phase`
- respect them literally
- implement the exact `next_attempt_blueprint`; do not pick a different safer mutation inside the same story
- if the idea needs a missing self-contained module in `train.py`, implement it instead of shrinking the idea into another precision nudge
- preserve causal scoring, exportability, reloadability, and honest byte accounting
- do not stack multiple hard ideas into one run unless the current idea explicitly calls for that combination
- disallowed attempt types unless they are only fixing a crash blocker in an otherwise valid new variant:
  - canonicalize, cleanup, restore, repair, or path-only commits
  - shifting float or QAT budget around within the same carrier
  - funding a same-family width tweak from a tiny Q compression
  - adding another copy of the same local module on the same outer carrier
  - branch-tip restore attempts that merely move back toward the current winner

Variant validity rule:
- A valid attempt must either:
  - introduce the named module or mechanism from the blueprint if it is not already present, or
  - change at least two macro axes from the accepted aggressive branch in the way the blueprint describes
- useful macro axes include:
  - shared vs tail partition
  - effective depth
  - `d_model`
  - `seq_len` bucket or curriculum shape
  - batch or grad-accum regime
  - local-module family
  - Q-rank topology
  - coarse quantization group or schedule
- if you cannot describe the attempt as a new architecture variant in one sentence, it is probably too small
- if the resulting carrier still looks like the current depth-3 neighbor-mixer line with a few toggles, it is too small

Validity rule:
- treat `ranking_policy` as the hard proxy-ranking rule for the campaign
- runs over the artifact hard cap or materially off the expected training budget are automatically non-contenders
- only accept runs that are ranking-valid
- the controller will reject an `accepted` decision for an invalid aggressive run, so do not try to promote one
- if a run is informative but invalid, revert it and record the lesson in notes
- if `recommended_phase` is still `establish`, do not spend a remaining attempt on a tiny cleanup or polish move

Free-form redesign rule:
- you are allowed to replace the accepted carrier wholesale
- you are allowed to change params, layers, partitioning, context, batch regime, and module stack
- you are allowed to write a new self-contained module in `train.py`
- large score swings, OOMs, and even clearly losing `1.60+` runs are acceptable telemetry if the redesign is real
- do not optimize for staying close to the current winner; optimize for honestly testing the blueprint

Attempt-style rule:
- attempts 1-4 should be the most different variants, not the safest ones
- do not spend attempts 1-4 on path cleanup, canonicalization, or same-carrier polish
- only use attempts 5-6 for refinement if one earlier variant is already a real contender
- otherwise attempts 5-6 should still be large alternative variants within the same idea

Useful framing for the first few attempts of an idea:
- attempt 1-2: establish a viable branch and get it training cleanly
- attempt 3-4: fix obvious weak points or spend saved compute/bytes more intelligently
- attempt 5-6: refine only if the branch looks real; otherwise take the biggest still-plausible variation within the same idea
- repeated invalid-budget or catastrophic runs are a sign to let the controller advance to the next story instead of forcing another polish attempt

Search policy for this aggressive loop:
- treat leaderboard patterns as priors, not recipes
- favor meaningful branch establishment over tiny score polishing
- the campaign default regime is roughly 15.2 MB to 15.9 MB, 11-12 layers, d_model 512, about 3x MLP, seq_len 2048, and batch around 524K tokens when feasible
- the active blueprint may deliberately step outside the current accepted branch shape; that is expected
- for XSA, cache, SmearGate, TTT, Canon, or smarter local-token stories, isolate the new idea enough that the result is interpretable
- if an idea naturally implies two families, keep them separate rather than blending them immediately
- do not spend low-rank-Q attempts only on same-carrier tail width or another copy of the same local module
- do not spend Canon or local-token attempts on canonicalizing an existing neighbor path
- do not spend XSA or cache attempts only on carrier prep without real XSA or cache behavior

Required loop for this invocation:
1. inspect the current aggressive branch tip, `$STATE_DIR/session.json`, `$STATE_DIR/aggressive_campaign.json`, `configs/aggressive_autoresearch_ideas.json`, `AGGRESSIVE_AUTORESEARCH_IDEAS.md`, `$STATE_DIR/notes.md`, and recent commits
2. read `next_attempt_blueprint`, restate it in your own words, and write down which macro axes it changes
3. edit `train.py`
4. commit the experiment, including the idea id in the commit message
5. run exactly one `bash scripts/run_autoresearch_experiment.sh`
6. inspect `latest.json`, `best.json`, and the concrete run outputs under `runs/autoresearch_5090_aggressive/`
7. if the run is ranking-valid and wins, keep the experiment commit; otherwise add a revert commit
8. record the decision with:
   `.venv/bin/python scripts/autoresearch_state.py --state_dir "$STATE_DIR" decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
9. record the attempt with:
   `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" record-attempt --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
10. if the run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`
11. update `$STATE_DIR/notes.md` with a short note about the tested hypothesis and outcome
12. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout or run.log as the control surface.
Use structured files and git history as memory.
```
