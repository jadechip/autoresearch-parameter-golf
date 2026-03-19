# Depth-Recurrent QAT GPT for Parameter Golf

This repo turns the original single-file reference into a production-shaped training, export, reload, and evaluation setup while keeping the core model/training logic centered in one editable file:

- `train.py`: main mutation target for model/training/search changes
- `prepare.py`: fixed smoke-data, tokenizer, and artifact-eval utilities
- `program.md`: instructions for external agents
- `validate_results.py`: fixed `results.json` schema validator
- `summarize_artifact.py`: submission-oriented artifact summary
- `train_pgolf_recurrent_qat.py`: compatibility wrapper for the old entrypoint

The baseline architecture is preserved by default:

- decoder-only LM
- tied embeddings
- grouped-query attention
- RoPE
- RMSNorm
- ReLU^2 MLP
- depth recurrence via shared blocks
- per-loop LoRA-style adapters
- fake INT8 quantization during training
- row-wise INT8 export with zlib packing
- optional linear weight averaging

## Install

Use a clean environment. The repo is pinned in `pyproject.toml`.

```bash
uv sync --extra dev --extra tokenizer
```

If you do not need real SentencePiece evaluation, `--extra tokenizer` is optional.

## Quick Start

Create tiny synthetic shards and a matching smoke config:

```bash
python prepare.py smoke-data --output_dir ./smoke_data
```

Run a fixed-time smoke experiment:

```bash
python train.py --config_json ./smoke_data/smoke_config.json --max_wallclock_seconds 30
```

Validate the structured output:

```bash
python validate_results.py ./smoke_data/run/results.json
```

Reload the exported artifact and evaluate it end-to-end:

```bash
python prepare.py eval-artifact \
  --config_json ./smoke_data/smoke_config.json \
  --artifact_path ./smoke_data/run/submission_bundle \
  --output_dir ./smoke_data/eval
```

Summarize submission-relevant bytes:

```bash
python summarize_artifact.py --results_json ./smoke_data/run/results.json
```

## Official Challenge Data

The upstream `openai/parameter-golf` repo documents an official cached FineWeb downloader for real training. This repo now mirrors that workflow locally through `prepare.py`.

Download the official `sp1024` challenge data and tokenizer into this repo's `./data/` directory:

```bash
bash scripts/download_official_fineweb.sh
```

For a smaller iteration subset:

```bash
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
```

Equivalent direct command:

```bash
uv run python prepare.py official-fineweb --variant sp1024 --train-shards 80
```

This creates the canonical layout expected by the training scripts:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

## Runpod Single-GPU Preset

Recommended GPU: `RTX 5090`.

Conservative ready-to-run preset files:

- `configs/runpod_5090_single_gpu.json`
- `scripts/runpod_5090_train.sh`

Example:

```bash
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
bash scripts/runpod_5090_train.sh
```

## 5090 Autoresearch Loop

For broad search on cheaper hardware, use the fixed-budget 5090 wrapper instead of long manual runs:

```bash
TRAIN_SHARDS=1 bash scripts/download_official_fineweb.sh
bash scripts/run_autoresearch_experiment.sh
```

This runs a single `300s` experiment with:

- a unique run directory under `./runs/autoresearch_5090/runs/`
- append-only `./runs/autoresearch_5090/results.tsv`
- stable agent-readable pointers:
  - `./runs/autoresearch_5090/index/latest.json`
  - `./runs/autoresearch_5090/index/best.json`

The intended flow is:

1. search on `1x5090`
2. promote only clear wins to `1xH100`
3. reserve `8xH100` for exact-budget submission rehearsals and record attempts

## Structured Outputs

Do not control runs by tailing raw logs. Consume these files instead:

- `output_dir/results.json`: fixed-schema result summary
- `output_dir/crash.json`: structured crash metadata on failure
- `results.tsv`: append-only experiment table
- `output_dir/submission_bundle/manifest.json`: export manifest and byte accounting
- `output_dir/checkpoints/latest.pt` and `final.pt`: resume points

The training script also prints grep-friendly final lines:

- `val_bpb:`
- `training_seconds:`
- `total_seconds:`
- `peak_vram_mb:`
- `mfu_percent:`
- `total_tokens_M:`
- `num_steps:`
- `num_params_M:`
- `depth:`
- `artifact_bytes:`

## Repo Notes

- `train.py` is intentionally the main editable file for external search loops.
- `prepare.py`, `validate_results.py`, `summarize_artifact.py`, and the docs are intended to stay stable.
- Export accounting snapshots the counted code files into `submission_bundle/code/` and counts those exact bytes plus the compressed model bytes.
- The default counted code set is just `train.py`, which keeps artifact accounting aligned with Parameter Golf’s preference for a single counted training file.
- Autonomous runs should be sandboxed and should read `results.json` / `crash.json`, not arbitrary stdout.

## Tests

CPU smoke:

```bash
pytest -q
```

Optional marker runs:

```bash
pytest -q -m cuda
pytest -q -m ddp
pytest -q -m compile_smoke
pytest -q -m sentencepiece_real
```
