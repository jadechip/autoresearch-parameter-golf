# Codex Aggressive Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for one aggressive idea-focused autoresearch iteration.

The outer supervisor launches a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, COMPETITIVE_PRIORS.md, AGGRESSIVE_AUTORESEARCH_IDEAS.md, and `$STATE_DIR/session.json` first.

You are running exactly one aggressive autoresearch iteration in this repo. Do not loop forever in this invocation.

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated aggressive autoresearch branch, create one yourself with a timestamped name like aggressive-autoresearch/20260325-010000
- inspect recent commits on that branch
- inspect `$STATE_DIR/session.json` and do not start unless it exists and `status=ready`
- inspect `$STATE_DIR/session.json["search_policy"]`, especially `lane`, `config_json`, `preflight`, `creative_guardrails`, `experiment_style`, and `refinement_gate`
- inspect `$STATE_DIR/aggressive_campaign.json`
- inspect the active idea via:
  `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" current`
- inspect `configs/aggressive_autoresearch_ideas.json`
- inspect `AGGRESSIVE_AUTORESEARCH_IDEAS.md`
- inspect `$STATE_DIR/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090_aggressive/index/latest.json`, `best.json`, and `best_raw.json` if they exist
- inspect the last several train-mode rows in `runs/autoresearch_5090_aggressive/results.tsv`

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision and campaign bookkeeping, stop

Discovery-first policy:
- this loop is for discovering viable branches, not for polishing forever
- use a stable carrier anchor whenever the lane policy provides one
- test one primary novelty axis at a time unless the active blueprint explicitly requires a combined test
- if `recommended_phase=repair`, do one clean budget or simplification repair instead of inventing a new story
- if `recommended_phase=refine`, only spend the attempt on one obvious local refinement that preserves attribution

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
- implement the exact `next_attempt_blueprint`, but keep it interpretable
- if the blueprint is too broad, honor its spirit with the smallest honest test that still matches the named mechanism
- do not stack multiple hard ideas into one run unless the blueprint explicitly calls for that combination
- keep public frontier patterns as priors, not recipes; preserve at least one clear creative difference in placement, schedule, carrier partition, or mechanism

Validity rule:
- a valid attempt must introduce the named mechanism or one clear budget repair for an already promising branch
- if you cannot explain the run in one sentence, it is too complicated
- if the run changes carrier topology, quantization story, and module family together, it is probably too complicated for this proxy

Preflight rule:
- `run_autoresearch_experiment.sh` may stop at a benchmark-only preflight
- the preflight now uses projected artifact bytes and derived throughput, not the raw benchmark score
- a benchmark-only `latest.json` with `preflight.decision=skip` counts as the completed experiment for this invocation
- treat that as informative telemetry, record why it failed, revert, and stop

Required loop for this invocation:
1. inspect branch tip, `$STATE_DIR/session.json`, `$STATE_DIR/aggressive_campaign.json`, `configs/aggressive_autoresearch_ideas.json`, `AGGRESSIVE_AUTORESEARCH_IDEAS.md`, `$STATE_DIR/notes.md`, recent commits, and the latest/best index files
2. read `next_attempt_blueprint`, `recommended_phase`, and `recommended_experiment_style`
3. restate the hypothesis in one sentence and explicitly name the one primary novelty axis
4. edit `train.py`
5. commit the experiment, including the idea id in the commit message
6. run exactly one `bash scripts/run_autoresearch_experiment.sh`
7. inspect `latest.json`, `best.json`, `best_raw.json`, and the concrete run outputs under `runs/autoresearch_5090_aggressive/`
8. if the run is a benchmark-only preflight reject, revert and record why it failed the gate
9. if the run is ranking-valid and wins, keep the experiment commit; otherwise add a revert commit
10. record the decision with:
   `.venv/bin/python scripts/autoresearch_state.py --state_dir "$STATE_DIR" decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
11. record the attempt with:
   `.venv/bin/python scripts/aggressive_autoresearch_campaign.py --state_dir "$STATE_DIR" record-attempt --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
12. if the run is accepted, commit the refreshed tracked files under `state/autoresearch/` and `configs/promoted/`
13. update `$STATE_DIR/notes.md` with a short note about the tested hypothesis, why it was chosen, and the outcome
14. stop after summarizing the one completed iteration

Do not start a second experiment in this invocation.
Do not rely on stdout as the control surface.
Use structured files and git history as memory.
```
