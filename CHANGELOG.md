# Changelog

## 0.1.0

- Promoted the repo from a single reference script to an autoresearch-shaped layout with `train.py`, `prepare.py`, `program.md`, pinned packaging, and stable helper scripts.
- Kept the baseline recurrent decoder architecture intact while making `train.py` the primary editable surface for external agents.
- Added structured run outputs: atomic `results.json`, `crash.json`, append-only `results.tsv`, and grep-friendly final metric lines.
- Added config serialization/hash support, stable run IDs, checkpoint save/resume, deterministic resume coverage, and fixed-time run handling.
- Replaced rough artifact-size estimation with exact submission-bundle accounting over counted code snapshots plus compressed model bytes.
- Added export manifesting, artifact reload/dequantize/model reconstruction, and end-to-end reload evaluation.
- Added benchmark reporting, results validation, and submission summary tooling.
- Expanded tests across config roundtrip, malformed shards, export/reload parity, validation parity, wall-clock stop, deterministic seeds, resume determinism, tokenizer mismatch, and optional CUDA/DDP/compile/SentencePiece markers.
