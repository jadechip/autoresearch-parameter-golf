# Production Readiness

## Completed

- [x] `train.py` is the main editable file for search loops.
- [x] Autoresearch-style repo shape exists: `prepare.py`, `train.py`, `program.md`, pinned packaging, and `results.tsv`.
- [x] Structured outputs exist: `results.json`, `crash.json`, export manifest, checkpoints, grep-friendly final metrics.
- [x] Atomic writes are used for structured JSON artifacts and compressed model output.
- [x] Checkpoint save/resume is implemented.
- [x] Config roundtrip and config hashing are implemented.
- [x] Fixed-time stopping is implemented and tested.
- [x] Export produces a counted submission bundle with exact byte accounting.
- [x] Exported artifacts can be reloaded and evaluated end-to-end.
- [x] Validation parity is checked between training-time export reload and separate evaluation path.
- [x] CPU smoke coverage is expanded.
- [x] Optional CUDA/DDP/compile/SentencePiece test markers exist.

## Remaining / Ongoing Risks

- [ ] GPU and DDP paths still depend on the local PyTorch/CUDA environment and should be exercised on the target hardware before long runs.
- [ ] `mfu_percent` is intentionally conservative (`0.0`) until a reliable cross-hardware estimator is added.
- [ ] Parameter Golf challenge accounting may still need final manual review against the latest official rules before submission.
- [ ] Real-tokenizer byte-accounting should be rechecked on the exact challenge tokenizer used for a final submission.
- [ ] Long-run restart behavior under preemption should be tested on the target job runner, not just CPU smoke tests.
