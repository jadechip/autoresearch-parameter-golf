# Codex Autoresearch Prompt

Use this as the bootstrap prompt for a Codex session running inside this repo on the 5090 host.

```text
Read program.md and AUTORESEARCH_SETUP.md first.

You are running an autoresearch loop on branch autoresearch/<tag>.

Current control files:
- runs/autoresearch_5090/index/latest.json
- runs/autoresearch_5090/index/best.json

Before making a new change:
- inspect the current branch
- inspect recent commits on the current branch
- inspect the current best.json

Primary goal:
- minimize val_bpb

Secondary constraints:
- preserve valid export and reloadability
- keep artifact_bytes under 16000000
- prefer changes that plausibly transfer to 1xH100 / 8xH100 production runs
- prefer simplicity

Rules:
- Edit only train.py.
- Do not modify prepare.py, helper scripts, docs, or dependencies.
- Run one experiment at a time with: bash scripts/run_autoresearch_experiment.sh
- Use runs/autoresearch_5090/index/latest.json and runs/autoresearch_5090/index/best.json as the control surface.
- Do not scrape stdout or rely on run.log.
- Architecture changes inside the current recurrent-QAT baseline family are allowed and encouraged.
- Do not chase 5090-only hacks that are unlikely to matter on H100.
- If artifact size remains far below the cap, consider bounded architecture changes instead of only micro-tuning optimizer settings.
- If a run fails, inspect crash.json and fix only if the idea still seems sound.
- If a run is worse or not meaningfully better, revert the train.py change and move on.
- If a run is a meaningful winner, commit the train.py change to the current branch.
- Include the run id and val_bpb in the commit message.
- Use recent git history as memory so you do not repeat the same weak ideas.
- Keep one experiment per code change.
- Keep the working state close to the current best candidate.
- Continue iterating autonomously until interrupted.
```

Recommended launch flow:

1. `git checkout autoresearch/<tag>`
2. `cat runs/autoresearch_5090/index/best.json`
3. `git log --oneline -n 10`
4. start Codex in the repo root
5. paste the prompt above
