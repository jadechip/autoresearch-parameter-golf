from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from train import (
    ModelConfig,
    TrainConfig,
    atomic_write_json,
    config_to_dict,
    evaluate_exported_artifact,
    load_sentencepiece_model,
    make_smoke_shards,
    train_config_from_dict,
)

OFFICIAL_MATCHED_FINEWEB_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
PREPARE_ROOT = Path(__file__).resolve().parent / "data"
DATASETS_DIR = PREPARE_ROOT / "datasets"
TOKENIZERS_DIR = PREPARE_ROOT / "tokenizers"


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def hf_resolve_url(repo_id: str, relative_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{relative_path}?download=1"


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return PREPARE_ROOT / remote_path


def download_remote_file(repo_id: str, relative_path: str) -> Path:
    destination = local_path_for_remote(relative_path)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(hf_resolve_url(repo_id, relative_path), headers={"User-Agent": "autoresearch-parameter-golf/0.1.0"})
    tmp_path = destination.with_name(f".{destination.name}.tmp")
    with urlopen(request) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(tmp_path, destination)
    return destination


def manifest_path() -> Path:
    return local_path_for_remote(f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/manifest.json")


def load_official_manifest(repo_id: str, *, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(f"manifest.json not present locally at {path}")
        download_remote_file(repo_id, f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts: list[str] = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed data prep, tokenizer, and artifact-eval utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke-data", help="Create tiny synthetic shards and a matching smoke config.")
    smoke.add_argument("--output_dir", type=str, required=True)
    smoke.add_argument("--vocab_size", type=int, default=32)
    smoke.add_argument("--train_tokens", type=int, default=512)
    smoke.add_argument("--val_tokens", type=int, default=128)

    fineweb = sub.add_parser(
        "official-fineweb",
        help="Download the official cached challenge FineWeb shards and tokenizer using the upstream Parameter Golf layout.",
    )
    fineweb.add_argument("--variant", type=str, default="sp1024")
    fineweb.add_argument("--train-shards", type=int, default=80)
    fineweb.add_argument("--skip-manifest", action="store_true")
    fineweb.add_argument("--with-docs", action="store_true")
    fineweb.add_argument("--repo-id", type=str, default=OFFICIAL_MATCHED_FINEWEB_REPO_ID)

    tok = sub.add_parser("tokenizer-info", help="Inspect a SentencePiece tokenizer.")
    tok.add_argument("--tokenizer_path", type=str, required=True)

    eval_art = sub.add_parser("eval-artifact", help="Reload an exported artifact and run validation.")
    eval_art.add_argument("--config_json", type=str, required=True)
    eval_art.add_argument("--artifact_path", type=str, required=True)
    eval_art.add_argument("--output_dir", type=str, default=None)
    eval_art.add_argument("--val_pattern", type=str, default=None)
    eval_art.add_argument("--tokenizer_path", type=str, default=None)

    return parser


def write_smoke_config(output_dir: Path, vocab_size: int) -> Path:
    cfg = TrainConfig(
        train_pattern=str(output_dir / "train_*.bin"),
        val_pattern=str(output_dir / "val_*.bin"),
        tokenizer_path=None,
        output_dir=str(output_dir / "run"),
        iterations=4,
        train_batch_tokens=64,
        val_batch_tokens=64,
        grad_accum_steps=1,
        log_every=1,
        val_every=1,
        max_wallclock_seconds=60.0,
        use_compile=False,
        use_lawa=False,
        checkpoint_every=1,
    )
    cfg.model = ModelConfig(
        vocab_size=vocab_size,
        seq_len=8,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        stem_layers=1,
        shared_layers=1,
        recurrence_loops=2,
        tail_layers=0,
        adapter_rank=2,
        adapter_alpha=4.0,
        fake_quant_during_train=True,
        fake_quant_start_step=0,
        emb_init_std=0.02,
    )
    path = output_dir / "smoke_config.json"
    atomic_write_json(path, config_to_dict(cfg))
    return path


def cmd_smoke_data(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = make_smoke_shards(
        output_dir,
        vocab_size=args.vocab_size,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
    )
    config_path = write_smoke_config(output_dir, args.vocab_size)
    print(json.dumps({"train_shard": str(train_path), "val_shard": str(val_path), "config_json": str(config_path)}, indent=2))
    return 0


def cmd_official_fineweb(args: argparse.Namespace) -> int:
    if args.train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    dataset_dir = dataset_dir_for_variant(args.variant)
    manifest = load_official_manifest(args.repo_id, skip_manifest_download=args.skip_manifest)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in {manifest_path()}")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if args.train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {args.repo_id}, requested {args.train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {manifest_path()}")

    prefix = f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    if args.with_docs:
        download_remote_file(args.repo_id, f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/docs_selected.jsonl")
        download_remote_file(args.repo_id, f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json")
    for idx in range(val_shards):
        download_remote_file(args.repo_id, f"{prefix}/fineweb_val_{idx:06d}.bin")
    for idx in range(args.train_shards):
        download_remote_file(args.repo_id, f"{prefix}/fineweb_train_{idx:06d}.bin")

    tokenizer_paths = [download_remote_file(args.repo_id, f"{OFFICIAL_MATCHED_FINEWEB_REMOTE_ROOT_PREFIX}/{path}") for path in artifact_paths_for_tokenizer(tokenizer_entry)]
    model_path = next((path for path in tokenizer_paths if path.suffix == ".model"), None)
    if model_path is None:
        raise ValueError(f"downloaded tokenizer artifacts did not include a .model file: {tokenizer_paths}")

    payload = {
        "repo_id": args.repo_id,
        "variant": args.variant,
        "train_shards": args.train_shards,
        "val_shards": val_shards,
        "dataset_dir": str(DATASETS_DIR / dataset_dir),
        "train_pattern": str(DATASETS_DIR / dataset_dir / "fineweb_train_*.bin"),
        "val_pattern": str(DATASETS_DIR / dataset_dir / "fineweb_val_*.bin"),
        "tokenizer_path": str(model_path),
        "manifest_path": str(manifest_path()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_tokenizer_info(args: argparse.Namespace) -> int:
    tokenizer = load_sentencepiece_model(args.tokenizer_path)
    print(json.dumps({"tokenizer_path": args.tokenizer_path, "vocab_size": int(tokenizer.vocab_size())}, indent=2))
    return 0


def cmd_eval_artifact(args: argparse.Namespace) -> int:
    cfg = train_config_from_dict(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    cfg.evaluate_only = True
    cfg.load_artifact_path = args.artifact_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.val_pattern is not None:
        cfg.val_pattern = args.val_pattern
    if args.tokenizer_path is not None:
        cfg.tokenizer_path = args.tokenizer_path
    summary = evaluate_exported_artifact(cfg)
    print(json.dumps(config_to_dict(summary), indent=2))
    return 0


def main() -> None:
    args = build_prepare_parser().parse_args()
    if args.cmd == "smoke-data":
        raise SystemExit(cmd_smoke_data(args))
    if args.cmd == "official-fineweb":
        raise SystemExit(cmd_official_fineweb(args))
    if args.cmd == "tokenizer-info":
        raise SystemExit(cmd_tokenizer_info(args))
    if args.cmd == "eval-artifact":
        raise SystemExit(cmd_eval_artifact(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
