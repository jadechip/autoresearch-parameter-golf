# Codex Minimal Autoresearch One-Shot Prompt

```text
You are running exactly one minimal autoresearch iteration in this repo. Do not loop forever in this invocation.

Read these first:
- minimal_autoresearch/README.md
- `$STATE_DIR/state.json`
- `.venv/bin/python minimal_autoresearch/state.py --state_dir "$STATE_DIR" show --recent 8`
- configs/aggressive_autoresearch_ideas.json
- `$AUTORESEARCH_ROOT/index/best_raw.json` if it exists
- the last several rows of `$AUTORESEARCH_ROOT/results.tsv` if it exists
- recent commits on the current branch

Goal:
- improve the architecture enough to get a lower validation bpb than the currently accepted run
- you may edit any model, training, config, or helper code needed for one coherent hypothesis
- use the aggressive-ideas file as a source of strong priors, not a rigid controller

Rules:
- stay on the current branch
- do exactly one experiment cycle
- make one coherent hypothesis, not a grab bag of unrelated changes
- you may edit multiple files if needed, but keep the story interpretable
- the evaluation protocol is frozen by `.minimal_autoresearch/state.json`; do not try to change wallclock budget, dataset/tokenizer inputs, validation mode, or other protected launch fields to win
- do not edit minimal_autoresearch/*.sh or minimal_autoresearch/state.py unless blocked by a clear bug in the minimal loop itself
- do not edit dependencies or dataset assets
- do not use git reset --hard
- use normal commits and normal revert commits only

Required steps:
1. Inspect the accepted baseline and recent attempts.
2. Pick one clear hypothesis, preferably inspired by one of the higher-priority ideas in configs/aggressive_autoresearch_ideas.json.
3. Edit the repo to implement that hypothesis.
4. Commit the experiment with a message starting with `mini-autoresearch:`.
5. Run exactly one experiment with `bash minimal_autoresearch/run_experiment.sh`.
6. Inspect `$AUTORESEARCH_ROOT/last_run.json`.
7. If a results json exists, run:
   `.venv/bin/python minimal_autoresearch/state.py --state_dir "$STATE_DIR" assess --results_json <results_json>`
8. If the assessment says `accept`, keep the experiment commit and record it with:
   `.venv/bin/python minimal_autoresearch/state.py --state_dir "$STATE_DIR" record --decision accepted --results_json <results_json> --experiment_commit <commit>`
9. Otherwise add a normal revert commit for the experiment and record it with:
   `.venv/bin/python minimal_autoresearch/state.py --state_dir "$STATE_DIR" record --decision reverted --results_json <results_json-if-any> --run_id <run_id-from-last_run_json-if-needed> --experiment_commit <commit> --revert_commit <revert_commit> --notes "<short reason>"`
10. Stop after summarizing the one completed iteration.

If training crashes or no results json is produced:
- revert the experiment commit
- read the `run_id` from `$AUTORESEARCH_ROOT/last_run.json`
- record a reverted attempt with `--run_id <run_id>` and a short note
- stop

Do not start a second experiment in this invocation.
Use the state file and results files as the control surface.
```
