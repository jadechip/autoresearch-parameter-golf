# Minimal Autoresearch

This is a stripped-down Codex loop for architecture search.

What it does:
- keeps a tiny accepted-state file in `.minimal_autoresearch/state.json`
- runs one Codex-guided experiment at a time
- trains exactly one candidate with `scripts/runpod_5090_train.sh`
- records every attempt in `.minimal_autoresearch/attempts.jsonl`
- freezes the evaluation protocol in state and enforces it at launch
- keeps the experiment commit if the run beats the current accepted result
- otherwise adds a normal revert commit

What it intentionally does not do:
- no aggressive campaign state machine
- no tracked accepted-state sync
- no preflight gate
- no multi-lane controller

Paths:
- state: `.minimal_autoresearch/`
- run outputs: `runs/minimal_autoresearch_5090/`
- loop scripts: `minimal_autoresearch/`

Quick start:

1. Initialize from a known good train result or index json:

```bash
.venv/bin/python minimal_autoresearch/state.py init \
  --baseline_results runs/autoresearch_5090/index/best_raw.json
```

2. Run one iteration:

```bash
bash minimal_autoresearch/run_once.sh
```

3. Run forever:

```bash
SKIP_UV_SYNC=1 bash minimal_autoresearch/run_forever.sh
```

Useful checks:

```bash
.venv/bin/python minimal_autoresearch/state.py show
jq . runs/minimal_autoresearch_5090/last_run.json
tail -n 20 runs/minimal_autoresearch_5090/results.tsv
```

Acceptance rule:
- candidate must be `status=success`
- candidate must be `mode=train`
- candidate must match the frozen protocol stored in `.minimal_autoresearch/state.json`
- candidate artifact must stay under the hard cap
- candidate train-loop time should stay within a reasonable ratio of the frozen wallclock budget
- candidate `val_bpb` must beat the current accepted `val_bpb` by at least the configured minimum improvement

Lower `val_bpb` is better.

Frozen protocol:
- the minimal loop does not let Codex change the evaluation budget, dataset/tokenizer inputs, validation mode, or other protected launch fields
- `run_experiment.sh` materializes a launch config with those protected fields enforced before training starts
- if a produced run drifts on protected fields, `state.py assess` rejects it with `protocol_drift`
