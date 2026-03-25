# External Agent Instructions

This repo uses a two-tier workflow:

- `1x5090` autoresearch is a short fixed-budget discovery tier.
- `1xH100` and `8xH100` runs are promotion / submission-rehearsal tiers.

Your job on the 5090 tier is not to fully win the final challenge in one run. Your job is to run an honest, information-dense proxy that helps discover candidate branches worth refining or promoting.

## Read First

At the start of a session, inspect:

- current git branch and recent git history on that branch
- `.autoresearch/session.json` or `.autoresearch_aggressive/session.json`
- `.autoresearch/notes.md` or `.autoresearch_aggressive/notes.md`
- `COMPETITIVE_PRIORS.md`
- `runs/autoresearch_5090/index/latest.json`
- `runs/autoresearch_5090/index/best.json`
- `runs/autoresearch_5090/index/best_raw.json` if it exists

Use git history plus structured results as the experiment memory.

## Accepted-State Policy

- Treat the current branch tip as the accepted code state.
- Treat recent git history as the canonical memory of accepted and rejected experiments.
- Treat `best.json` and `best_valid.json` as ranking-valid-only pointers.
- Treat `best_raw.json` as numeric telemetry only; it may point at a reverted or invalid run.
- Do not let any index file override the accepted branch state on its own.
- Treat `.autoresearch/session.json` as the readiness gate. If it is missing or not `ready`, do not start the loop.

## What You May Edit

- `train.py`
- `.autoresearch/notes.md` or `.autoresearch_aggressive/notes.md`
- a minimal helper script only when a hypothesis cannot be tested honestly without preserving structured outputs and export/reload invariants
- lane policy or idea files when the search process itself is clearly wasting runs

## What You Must Not Rely On

Do not treat terminal output as the control surface. Use structured files:

- `output_dir/results.json`
- `output_dir/crash.json`
- `results.tsv`
- `output_dir/submission_bundle/manifest.json`
- `runs/.../index/latest.json`
- `runs/.../index/best.json`
- `runs/.../index/best_raw.json`
- `.autoresearch/session.json`
- `.autoresearch/experiments.jsonl`
- `.autoresearch_aggressive/aggressive_campaign.json`

## Non-Negotiable Invariants

- preserve tokenizer-agnostic bits-per-byte correctness
- keep `results.json` schema valid
- keep export -> reload -> eval working for full runs
- keep checkpoint save/resume working
- crash fast on invalid configs
- keep `train.py` as the main mutation target
- preserve causal scoring and honest artifact accounting
- keep `artifact_bytes < 16,000,000`

## First-Principles 5090 Search Strategy

Treat the 5090 as a noisy but useful ranking proxy.

The proxy is strongest when:

- the carrier anchor is stable enough that run-to-run comparisons mean something
- one primary novelty is tested at a time
- preflight kills obvious byte/throughput losers before full runs
- invalid near-misses trigger budget repair, not another totally new story
- dead families are demoted quickly instead of consuming all remaining attempts

In practice, this means:

- start from the lane policy’s `config_json` when one is provided
- prefer a stable near-frontier seed carrier for the `frontier_pure_model` lane
- allow at most one primary novelty plus one small support change per 5-minute proxy
- do not combine topology + quantization + curriculum + module family in one run unless the blueprint explicitly demands it
- treat 1024-1536 context as conditional on preserved throughput; do not assume 2048 is the default 5090 proxy regime

## Discovery Versus Refinement

Aggressive loop:

- discovery-first
- establish viable branch families
- once a branch is ranking-valid and close enough, hand it off instead of polishing forever
- use `recommended_phase` from the aggressive campaign: `establish`, `repair`, or `refine`

Ordinary loop:

- refinement-first
- operate on ranking-valid contenders
- prefer post-quant selection, checkpoint choice, carrier-local cleanup, and one-knob refinements

## Search Priorities

Highest EV in this repo right now:

- stable near-cap 11-12 layer carriers
- late selective quantization and PTQ/post-quant checkpoint choice
- low-rank Q as a single-axis reallocation tool
- activation / RoPE / residual-scale variants local to `train.py`
- localized Canon-style inserts on otherwise stable carriers

Lower EV unless strong new evidence appears:

- blanket or headline INT4 MLP sweeps
- heavyweight local-token modules
- many-axis moonshots in one 5-minute proxy
- returning to tiny compact recurrent families as the default frontier direction

## Git Discipline

- make one commit for the experiment before running it
- if a run loses, fails, or is rejected at preflight, preserve it in history and return to accepted state with a normal revert commit
- do not use `git reset` to erase experiment history during the loop
- do not accumulate multiple speculative edits without a run in between
- update session state and notes after each keep/revert decision

Among candidates with similar `val_bpb`, prefer:

1. simpler code
2. ranking-valid results
3. deliberate byte use near the target band
4. lower VRAM and cleaner throughput
