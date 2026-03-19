# Codex Autoresearch Prompt

Use this as the bootstrap prompt for an interactive Codex session running inside this repo on the 5090 host.

For overnight / continuously restarted execution, prefer:

```bash
bash scripts/run_codex_autoresearch_loop.sh
```

That supervisor uses `CODEX_AUTORESEARCH_ONE_SHOT_PROMPT.md` and launches a fresh `codex exec` for each iteration.

Recommended human launch flow:

1. `cd /workspace/autoresearch-parameter-golf`
2. make sure the lightweight session gate exists:
   `bash scripts/init_autoresearch_session.sh`
3. start Codex in the repo root with:
   `codex --dangerously-bypass-approvals-and-sandbox`
   Only do this on a dedicated remote box you control for these experiments.
4. paste the prompt below

```text
Read program.md and AUTORESEARCH_SETUP.md first.

You are running an autoresearch loop in this repo.

Before doing anything else:
- inspect the current git branch
- if you are not already on a dedicated autoresearch branch, create one yourself with a timestamped name like autoresearch/20260319-153000
- inspect recent commits on that branch
- inspect .autoresearch/session.json and do not start experimenting unless it exists and status=ready
- inspect `.autoresearch/session.json["search_policy"]` and `.autoresearch/notes.md`
- inspect runs/autoresearch_5090/index/best.json

Git policy for this session:
- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Make one commit for every experiment attempt before running it.
- If an experiment loses or crashes, preserve it in history and return to the accepted state with a normal revert commit.
- Do not use git reset to erase experiment history.

Current control files:
- runs/autoresearch_5090/index/latest.json
- runs/autoresearch_5090/index/best.json
- .autoresearch/session.json
- .autoresearch/experiments.jsonl
- .autoresearch/notes.md

Before making a new change:
- inspect the current best.json again if needed
- inspect the current hypothesis ledger in `.autoresearch/notes.md`

Primary goal:
- minimize val_bpb

Secondary constraints:
- preserve valid export and reloadability
- keep artifact_bytes under 16000000
- prefer changes that plausibly transfer to 1xH100 / 8xH100 production runs
- prefer simplicity

Rules:
- Edit only `train.py` and the local session file `.autoresearch/notes.md`.
- Do not modify prepare.py, helper scripts, docs, or dependencies.
- Run one experiment at a time with: bash scripts/run_autoresearch_experiment.sh
- Use runs/autoresearch_5090/index/latest.json and runs/autoresearch_5090/index/best.json as the control surface.
- Do not scrape stdout or rely on run.log.
- Treat runs/autoresearch_5090/index/best.json as numeric telemetry, not the sole source of truth for accepted state.
- Architecture changes inside the current recurrent-QAT baseline family are allowed and encouraged.
- Do not chase 5090-only hacks that are unlikely to matter on H100.
- If accepted artifact size remains below the search-policy target band, prefer bounded architecture and byte-allocation experiments over endless optimizer micro-tuning.
- The first search block in a fresh session should cover structural axes before long micro-tuning streaks: `d_model`, `shared_layers` vs `recurrence_loops`, `tail_layers`, `mlp_mult`, `adapter_rank` / `adapter_targets`, and fake-quant timing / clip percentile.
- Do not spend more than the search-policy limit of consecutive losing micro-tuning experiments without making the next run a structural or byte-allocation experiment.
- Use `.autoresearch/notes.md` as a durable ledger of open, tried, winning, and rejected structural hypotheses.
- If a run fails, inspect crash.json and fix only if the idea still seems sound.
- Before each run, commit the exact train.py experiment to the current autoresearch branch.
- Include the experiment idea in the pre-run commit message.
- If a run is worse or not meaningfully better, add a revert commit and move on.
- If a run is a meaningful winner, keep the experiment commit as part of the accepted branch history.
- Include the run id and val_bpb in the post-run keep/revert commit message when possible.
- After each keep/revert decision, update the Ralph-style state file with:
  `uv run python scripts/autoresearch_state.py --state_dir ./.autoresearch decide --run_id <run_id> --decision accepted|reverted --results_json <results_json>`
- After each keep/revert decision, update `.autoresearch/notes.md` with a short note about what hypothesis was tested and whether it won, lost, or remains unresolved.
- Use recent git history as memory so you do not repeat the same weak ideas.
- Keep one experiment per code change.
- Keep the branch tip equal to the current accepted candidate.
- Continue iterating autonomously until interrupted.
```

Expected command loop inside Codex:

1. inspect branch tip, `.autoresearch/session.json`, `.autoresearch/notes.md`, recent commits, and `runs/autoresearch_5090/index/best.json`
2. edit `train.py`
3. commit the experiment
4. run `bash scripts/run_autoresearch_experiment.sh`
5. inspect `latest.json`, `best.json`, and the concrete run outputs
6. keep the commit if it wins, otherwise add a revert commit
7. record the decision in `.autoresearch/session.json` and update `.autoresearch/notes.md`
8. continue
