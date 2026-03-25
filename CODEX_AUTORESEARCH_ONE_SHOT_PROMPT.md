# Codex Autoresearch One-Shot Prompt

Use this prompt with `codex exec` for a single refinement-oriented autoresearch iteration.

The outer supervisor should launch a fresh Codex process for each iteration, so this prompt must do exactly one experiment cycle and then stop.

```text
Read program.md, AUTORESEARCH_SETUP.md, COMPETITIVE_PRIORS.md, and `.autoresearch/session.json` first.

You are running exactly one autoresearch iteration in this repo. Do not loop forever in this invocation.

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated autoresearch branch, create one yourself with a timestamped name like autoresearch/20260325-010000
- inspect recent commits on that branch
- run `.venv/bin/python scripts/summarize_recent_autoresearch.py --limit 12`
- inspect `.autoresearch/session.json` and do not start unless it exists and `status=ready`
- inspect `.autoresearch/session.json["search_policy"]`, including `lane`, `config_json`, `preflight`, `discovery_mode`, `experiment_style`, and `refinement_gate`
- inspect `.autoresearch/notes.md`
- inspect `COMPETITIVE_PRIORS.md`
- inspect `runs/autoresearch_5090/index/latest.json`, `best.json`, and `best_raw.json`

Execution model for this invocation:
- perform at most one code change
- run at most one experiment with `bash scripts/run_autoresearch_experiment.sh`
- after the keep/revert decision, stop

Controller model:
- the aggressive loop should establish families; this loop should refine real contenders
- use the lane policy’s `config_json` as the stable anchor unless the hypothesis explicitly requires a different starting point
- prefer one local refinement or one post-quant / checkpoint-selection move per run
- if there is no real contender, do not pretend there is one; switch back to a structural story instead of polishing noise

Git policy:
- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Make one commit for the experiment before running it.
- If the run loses, fails, or is rejected at preflight, preserve it in history and return to accepted state with a normal revert commit.
- Do not use `git reset` to erase experiment history.

Allowed files:
- edit `train.py`
- update `.autoresearch/notes.md`
- add one minimal helper-script change only if the refinement cannot be tested honestly without preserving structured outputs

Do not modify:
- `prepare.py`
- dependencies
- unrelated docs

Search policy:
- treat `best.json` as ranking-valid-only telemetry
- treat `best_raw.json` as raw numeric telemetry only
- prefer ranking-valid contenders under the cap over oversize near-misses
- default to one primary novelty or one local refinement axis per run
- if accepted artifact size is below the target band, use bounded byte-allocation moves instead of endless optimizer micro-tuning
- if recent history is narrow, force a different branch family before another local refinement

Required loop for this invocation:
1. inspect branch tip, `.autoresearch/session.json`, `.autoresearch/notes.md`, `COMPETITIVE_PRIORS.md`, recent commits, `best.json`, `best_raw.json`, and the recent summary
2. decide whether this iteration is contender refinement or whether the loop should return to structural exploration
3. choose one concrete next hypothesis and name the one primary axis it changes
4. edit `train.py`
5. commit the experiment
6. run exactly one `bash scripts/run_autoresearch_experiment.sh`
7. inspect `latest.json`, `best.json`, `best_raw.json`, and the concrete run outputs
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
