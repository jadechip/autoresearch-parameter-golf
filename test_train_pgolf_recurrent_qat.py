from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import compare_runs
import train as mod
import watch_run

_AUTORESEARCH_STATE_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_state",
    Path(__file__).parent / "scripts" / "autoresearch_state.py",
)
assert _AUTORESEARCH_STATE_SPEC is not None and _AUTORESEARCH_STATE_SPEC.loader is not None
autoresearch_state = importlib.util.module_from_spec(_AUTORESEARCH_STATE_SPEC)
_AUTORESEARCH_STATE_SPEC.loader.exec_module(autoresearch_state)


class FakeTokenizer:
    def __init__(self, vocab_size: int = 6):
        self._pieces = {
            0: "<unk>",
            1: "▁hi",
            2: "there",
            3: "<ctrl>",
            4: "<unused>",
            5: "<byte>",
        }
        self._vocab_size = vocab_size

    def vocab_size(self) -> int:
        return self._vocab_size

    def is_control(self, token_id: int) -> bool:
        return token_id == 3

    def is_unknown(self, token_id: int) -> bool:
        return token_id == 0

    def is_unused(self, token_id: int) -> bool:
        return token_id == 4

    def is_byte(self, token_id: int) -> bool:
        return token_id == 5

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]


def tiny_cfg(tmp_path: Path, *, name: str = "run") -> mod.TrainConfig:
    cfg = mod.TrainConfig(
        train_pattern=str(tmp_path / "train_*.bin"),
        val_pattern=str(tmp_path / "val_*.bin"),
        tokenizer_path=None,
        output_dir=str(tmp_path / name),
        results_tsv_path=str(tmp_path / "results.tsv"),
        iterations=4,
        train_batch_tokens=64,
        val_batch_tokens=64,
        grad_accum_steps=1,
        log_every=1,
        val_every=1,
        checkpoint_every=1,
        max_wallclock_seconds=60.0,
        use_compile=False,
        use_lawa=False,
    )
    cfg.model = mod.ModelConfig(
        vocab_size=32,
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
    cfg.optim = mod.OptimConfig(
        embed_lr=3e-3,
        head_lr=3e-3,
        matrix_lr=1e-2,
        scalar_lr=3e-3,
        grad_clip_norm=1.0,
        warmup_steps=1,
        warmdown_steps=2,
    )
    return cfg


def make_easy_shards(tmp_path: Path, vocab_size: int = 32, n_train: int = 512, n_val: int = 128) -> None:
    pattern = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    train = np.tile(pattern, n_train // len(pattern) + 1)[:n_train] % vocab_size
    val = np.tile(pattern, n_val // len(pattern) + 1)[:n_val] % vocab_size
    mod.write_data_shard(tmp_path / "train_000.bin", train)
    mod.write_data_shard(tmp_path / "val_000.bin", val)


def load_checkpoint_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_state_dict_equal(lhs: dict, rhs: dict) -> None:
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        left = lhs[key]
        right = rhs[key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), key
        else:
            assert left == right, key


def write_fake_results(path: Path, *, run_id: str, val_bpb: float, status: str = "success") -> Path:
    run_dir = path.parent
    payload = {
        "schema_version": mod.RESULTS_SCHEMA_VERSION,
        "status": status,
        "mode": "train",
        "run_id": run_id,
        "config_hash": f"hash-{run_id}",
        "output_dir": str(run_dir),
        "results_path": str(path),
        "config_path": str(run_dir / "config.json"),
        "started_at_unix": 0.0,
        "finished_at_unix": 1.0,
        "training_seconds": 1.0,
        "total_seconds": 1.0,
        "peak_vram_mb": 0.0,
        "mfu_percent": 0.0,
        "total_tokens": 1,
        "total_tokens_M": 0.000001,
        "num_steps": 1,
        "num_params": 1,
        "num_params_M": 0.000001,
        "depth": 1,
        "train_loss": 1.0,
        "val_loss": 1.0,
        "val_bpb": val_bpb,
        "artifact_bytes": 1,
        "artifact": None,
        "benchmark": None,
        "checkpoint_path": None,
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "tensorboard_log_dir": str(run_dir / "tensorboard"),
        "resume_from": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_config_roundtrip_json_dataclass_json(tmp_path: Path) -> None:
    cfg = tiny_cfg(tmp_path)
    payload = mod.config_to_dict(cfg)
    roundtrip = mod.train_config_from_dict(json.loads(json.dumps(payload)))
    assert mod.config_to_dict(roundtrip) == payload
    assert mod.config_hash(cfg) == mod.config_hash(roundtrip)


def test_validation_schedule_helpers() -> None:
    assert mod.should_run_validation(step=0, target_iterations=100, val_every=0, eval_first_step=False) is False
    assert mod.should_run_validation(step=0, target_iterations=100, val_every=100, eval_first_step=True) is True
    assert mod.should_run_validation(step=50, target_iterations=100, val_every=0, eval_first_step=False) is False
    assert mod.should_run_validation(step=99, target_iterations=100, val_every=0, eval_first_step=False) is True
    assert mod.should_run_post_loop_validation(True, completed_steps=12, last_step=11, last_val_step=None) is True
    assert mod.should_run_post_loop_validation(True, completed_steps=12, last_step=11, last_val_step=11) is False


def test_write_and_load_shard_roundtrip(tmp_path: Path) -> None:
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    shard = mod.write_data_shard(tmp_path / "toy.bin", tokens)
    loaded = mod.load_data_shard(shard)
    assert loaded.dtype == torch.uint16
    assert loaded.tolist() == tokens.tolist()


def test_malformed_shard_header_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        mod.load_data_shard(bad)


def test_token_stream_wraps_across_files(tmp_path: Path) -> None:
    mod.write_data_shard(tmp_path / "a.bin", [1, 2, 3])
    mod.write_data_shard(tmp_path / "b.bin", [4, 5, 6])
    stream = mod.TokenStream(str(tmp_path / "*.bin"))
    taken = stream.take(8)
    assert taken.tolist() == [1, 2, 3, 4, 5, 6, 1, 2]


def test_build_piece_byte_luts_fake_tokenizer() -> None:
    base, has_space, is_boundary = mod.build_piece_byte_luts(FakeTokenizer(), vocab_size=8, device=torch.device("cpu"))
    assert int(base[1].item()) == len("hi".encode("utf-8"))
    assert bool(has_space[1].item()) is True
    assert bool(is_boundary[1].item()) is False
    assert int(base[5].item()) == 1


def test_cuda_metric_helpers_are_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", 0)
    monkeypatch.setattr(mod.torch.cuda, "set_device", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad set_device")))
    monkeypatch.setattr(
        mod.torch.cuda,
        "reset_peak_memory_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad reset")),
    )
    monkeypatch.setattr(
        mod.torch.cuda,
        "max_memory_allocated",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad alloc")),
    )
    mod.reset_peak_vram_stats(device)
    assert mod.peak_vram_mb(device) == 0.0


def test_setup_seed_deterministic_configures_cuda_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod.torch.cuda, "manual_seed_all", lambda seed: calls.append(("manual_seed_all", seed)))
    monkeypatch.setattr(mod.torch, "use_deterministic_algorithms", lambda *args, **kwargs: calls.append(("det", (args, kwargs))))
    monkeypatch.setattr(mod.torch.backends.cuda, "enable_flash_sdp", lambda flag: calls.append(("flash", flag)))
    monkeypatch.setattr(mod.torch.backends.cuda, "enable_mem_efficient_sdp", lambda flag: calls.append(("mem", flag)))
    monkeypatch.setattr(mod.torch.backends.cuda, "enable_math_sdp", lambda flag: calls.append(("math", flag)))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mod.setup_seed(123, deterministic=True)
    assert ("flash", False) in calls
    assert ("mem", False) in calls
    assert ("math", True) in calls


def test_rope_and_attention_shapes() -> None:
    cfg = mod.ModelConfig(vocab_size=32, seq_len=8, d_model=32, num_heads=4, num_kv_heads=2, shared_layers=1, recurrence_loops=2)
    attn = mod.GroupedQueryAttention(cfg, num_adapter_slots=cfg.recurrence_loops)
    x = torch.randn(2, 8, 32)
    y = attn(x, slot=1)
    assert y.shape == x.shape
    rope = mod.RotaryEmbedding(dim=8)
    cos, sin = rope.get_cos_sin(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (8, 4)
    assert sin.shape == (8, 4)


def test_fake_quant_linear_adapter_slot_changes_output() -> None:
    layer = mod.FakeQuantLinear(
        8,
        8,
        bias=False,
        fake_quant_during_train=True,
        fake_quant_start_step=0,
        num_adapter_slots=2,
        adapter_rank=2,
        adapter_alpha=2.0,
    )
    with torch.no_grad():
        layer.adapter_A[1].fill_(0.5)
        layer.adapter_B[1].fill_(0.25)
    x = torch.randn(2, 3, 8)
    out0 = layer(x, slot=0)
    out1 = layer(x, slot=1)
    assert out0.shape == out1.shape == (2, 3, 8)
    assert not torch.allclose(out0, out1)


def test_model_forward_and_loss_finite() -> None:
    cfg = mod.ModelConfig(
        vocab_size=32,
        seq_len=8,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        stem_layers=1,
        shared_layers=2,
        recurrence_loops=2,
        tail_layers=1,
        adapter_rank=2,
    )
    model = mod.RecurrentGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, loss = model(x, y)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert loss is not None and math.isfinite(float(loss.item()))
    assert cfg.effective_depth == 1 + 2 * 2 + 1


def test_quantization_pack_roundtrip() -> None:
    cfg = mod.ModelConfig(vocab_size=32, seq_len=8, d_model=32, num_heads=4, num_kv_heads=2, stem_layers=1, shared_layers=1, recurrence_loops=2, adapter_rank=2)
    model = mod.RecurrentGPT(cfg)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload, stats = mod.quantize_state_dict_int8(state, mod.QuantConfig())
    assert stats["param_count"] > 0
    blob = mod.pack_quantized_payload(payload, mod.QuantConfig())
    restored_payload = mod.unpack_quantized_payload(blob)
    restored = mod.dequantize_state_dict_int8(restored_payload)
    for key, tensor in state.items():
        assert key in restored
        assert restored[key].shape == tensor.shape
        if tensor.is_floating_point() and tensor.numel() > mod.QuantConfig().keep_float_max_numel:
            mae = (restored[key].float() - tensor.float()).abs().mean().item()
            assert mae < 0.1


def test_export_reload_forward_pass_parity(tmp_path: Path) -> None:
    cfg = tiny_cfg(tmp_path)
    model = mod.RecurrentGPT(cfg.model)
    export_stats, _manifest = mod.export_quantized_artifact(
        cfg=cfg,
        model=model,
        output_dir=Path(cfg.output_dir),
        run_id="test-run",
        cfg_hash=mod.config_hash(cfg),
    )
    loaded_model, _loaded_manifest, _artifact_dir = mod.load_model_from_artifact(Path(export_stats.artifact_dir), torch.device("cpu"))
    payload = mod.unpack_quantized_payload(Path(export_stats.model_blob_path).read_bytes())
    manual_model = mod.RecurrentGPT(cfg.model)
    manual_model.load_state_dict(mod.dequantize_state_dict_int8(payload), strict=True)
    x = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.seq_len))
    logits_loaded, _ = loaded_model(x)
    logits_manual, _ = manual_model(x)
    assert torch.allclose(logits_loaded, logits_manual, atol=1e-6, rtol=0.0)


def test_training_step_decreases_loss_on_easy_sequence() -> None:
    cfg = mod.ModelConfig(vocab_size=16, seq_len=8, d_model=32, num_heads=4, num_kv_heads=2, stem_layers=1, shared_layers=1, recurrence_loops=2, tail_layers=0, adapter_rank=2)
    model = mod.RecurrentGPT(cfg)
    train_cfg = mod.TrainConfig(iterations=16)
    train_cfg.model = cfg
    train_cfg.optim = mod.OptimConfig(embed_lr=2e-3, head_lr=2e-3, matrix_lr=8e-3, scalar_lr=2e-3, warmup_steps=1, warmdown_steps=4)
    bundle = mod.build_optimizers(model, train_cfg)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1], [2, 3, 4, 5, 6, 7, 8, 1]], dtype=torch.long)
    losses = []
    for step in range(12):
        model.set_global_step(step)
        mod.set_optimizer_lrs(bundle, step, 12, train_cfg.optim)
        for opt in bundle.all():
            opt.zero_grad(set_to_none=True)
        _logits, loss = model(x, y)
        assert loss is not None
        loss.backward()
        for opt in bundle.all():
            opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]


def test_full_smoke_train_and_export(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    summary = mod.train_one_run(cfg)
    results = mod.load_and_validate_results(Path(cfg.output_dir) / "results.json")
    assert summary.model_params > 0
    assert summary.effective_depth == cfg.model.effective_depth
    assert math.isfinite(summary.train_loss or 0.0)
    assert summary.val_loss is not None and math.isfinite(summary.val_loss)
    assert summary.export is not None
    assert (Path(cfg.output_dir) / cfg.artifact_bundle_name / "model_int8.zlib").exists()
    assert summary.export.artifact_bytes > 0
    assert Path(summary.export.manifest_path).exists()
    assert results["artifact_bytes"] == summary.export.artifact_bytes
    assert summary.export.reload_val_loss is not None
    if summary.val_bpb is None:
        assert summary.export.reload_val_bpb is None
    else:
        assert summary.export.reload_val_bpb is not None
        assert abs(summary.export.reload_val_bpb - summary.val_bpb) < 0.05


def test_export_reload_validation_parity(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    train_cfg = tiny_cfg(tmp_path, name="train")
    train_summary = mod.train_one_run(train_cfg)
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg.output_dir = str(tmp_path / "eval")
    eval_cfg.evaluate_only = True
    eval_cfg.load_artifact_path = str(Path(train_cfg.output_dir) / train_cfg.artifact_bundle_name)
    eval_summary = mod.evaluate_exported_artifact(eval_cfg)
    assert train_summary.export is not None
    if train_summary.export.reload_val_bpb is None:
        assert eval_summary.val_bpb is None
    else:
        assert abs(eval_summary.val_bpb - train_summary.export.reload_val_bpb) < 1e-6
    assert train_summary.export.reload_val_loss is not None
    assert abs(eval_summary.val_loss - train_summary.export.reload_val_loss) < 1e-6


def test_fixed_time_early_stop_behavior(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.iterations = 100
    cfg.max_wallclock_seconds = 0.0
    summary = mod.train_one_run(cfg)
    assert summary.num_steps == 0
    assert summary.total_tokens == 0
    assert summary.step == -1


def test_benchmark_mode_reports_tokens_per_second(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.benchmark_only = True
    cfg.benchmark_train_steps = 2
    cfg.save_final_quantized = False
    summary = mod.train_one_run(cfg)
    assert summary.mode == "benchmark"
    assert summary.benchmark is not None
    assert summary.benchmark.train_tokens_per_second > 0.0


def test_deterministic_seed_behavior_short_cpu_run(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg_a = tiny_cfg(tmp_path / "a_data", name="run_a")
    make_easy_shards(tmp_path / "a_data")
    cfg_b = tiny_cfg(tmp_path / "b_data", name="run_b")
    make_easy_shards(tmp_path / "b_data")
    cfg_b.seed = cfg_a.seed
    summary_a = mod.train_one_run(cfg_a)
    summary_b = mod.train_one_run(cfg_b)
    ckpt_a = load_checkpoint_state(Path(cfg_a.output_dir) / "checkpoints" / "final.pt")
    ckpt_b = load_checkpoint_state(Path(cfg_b.output_dir) / "checkpoints" / "final.pt")
    assert summary_a.val_bpb == summary_b.val_bpb
    assert ckpt_a["next_step"] == ckpt_b["next_step"]
    assert_state_dict_equal(ckpt_a["model_state"], ckpt_b["model_state"])


def test_checkpoint_resume_deterministic(tmp_path: Path) -> None:
    make_easy_shards(tmp_path / "full_data")
    cfg_full = tiny_cfg(tmp_path / "full_data", name="full_run")
    cfg_full.iterations = 4
    cfg_full.use_lawa = False
    full_summary = mod.train_one_run(cfg_full)

    make_easy_shards(tmp_path / "resume_data")
    cfg_part = tiny_cfg(tmp_path / "resume_data", name="resume_run")
    cfg_part.iterations = 2
    cfg_part.use_lawa = False
    mod.train_one_run(cfg_part)

    cfg_resume = tiny_cfg(tmp_path / "resume_data", name="resume_run")
    cfg_resume.iterations = 4
    cfg_resume.use_lawa = False
    cfg_resume.resume_from = str(Path(cfg_part.output_dir) / "checkpoints" / "final.pt")
    resume_summary = mod.train_one_run(cfg_resume)

    full_ckpt = load_checkpoint_state(Path(cfg_full.output_dir) / "checkpoints" / "final.pt")
    resume_ckpt = load_checkpoint_state(Path(cfg_resume.output_dir) / "checkpoints" / "final.pt")
    assert full_summary.val_bpb == resume_summary.val_bpb
    assert_state_dict_equal(full_ckpt["model_state"], resume_ckpt["model_state"])


def test_tokenizer_vocab_mismatch_error_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.tokenizer_path = str(tmp_path / "fake.model")
    monkeypatch.setattr(mod, "load_sentencepiece_model", lambda _path: FakeTokenizer(vocab_size=999))
    with pytest.raises(ValueError, match="vocab size mismatch"):
        mod.train_one_run(cfg)


def test_results_validator_script(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    mod.train_one_run(cfg)
    proc = subprocess.run(
        [sys.executable, "validate_results.py", str(Path(cfg.output_dir) / "results.json")],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"validated": true' in proc.stdout.lower()


def test_metrics_jsonl_written_for_training_run(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.metrics_jsonl_path = "metrics.jsonl"
    mod.train_one_run(cfg)
    metrics_path = Path(cfg.output_dir) / "metrics.jsonl"
    tensorboard_dir = Path(cfg.output_dir) / "tensorboard"
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event["event"] == "run_start" for event in events)
    assert any(event["event"] == "train" for event in events)
    assert any(event["event"] == "val" for event in events)
    assert events[-1]["event"] == "summary"
    train_event = next(event for event in events if event["event"] == "train")
    assert train_event["matrix_lr"] > 0.0
    results = json.loads((Path(cfg.output_dir) / "results.json").read_text(encoding="utf-8"))
    assert results["metrics_path"] == str(metrics_path)
    assert results["tensorboard_log_dir"] == str(tensorboard_dir)
    assert list(tensorboard_dir.glob("events.out.tfevents.*"))


def test_watch_run_script_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    assert watch_run.parse_metric_names("matrix_lr,val_bpb,tokens_per_second") == ("matrix_lr", "val_bpb", "tokens_per_second")
    metrics_events = [
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "run_start", "run_id": "demo", "mode": "train"},
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "train", "step": 0, "train_loss": 5.0, "matrix_lr": 0.01, "step_seconds": 1.0, "total_tokens": 64, "elapsed_training_seconds": 2.0},
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "val", "phase": "final", "step": 0, "val_loss": 4.5, "val_bpb": 2.0},
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "summary", "status": "success", "step": 0, "num_steps": 1, "val_bpb": 2.0},
    ]
    metrics_path.write_text("\n".join(json.dumps(event) for event in metrics_events) + "\n", encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "schema_version": mod.RESULTS_SCHEMA_VERSION,
                "status": "success",
                "mode": "train",
                "run_id": "demo",
                "config_hash": "hash-demo",
                "output_dir": str(run_dir),
                "results_path": str(run_dir / "results.json"),
                "config_path": str(run_dir / "config.json"),
                "metrics_path": str(metrics_path),
                "started_at_unix": 0.0,
                "finished_at_unix": 1.0,
                "training_seconds": 1.0,
                "total_seconds": 1.0,
                "peak_vram_mb": 0.0,
                "mfu_percent": 0.0,
                "total_tokens": 1,
                "total_tokens_M": 0.000001,
                "num_steps": 1,
                "num_params": 1,
                "num_params_M": 0.000001,
                "depth": 1,
                "train_loss": 5.0,
                "val_loss": 4.5,
                "val_bpb": 2.0,
                "artifact_bytes": 1,
                "artifact": None,
                "benchmark": None,
                "checkpoint_path": None,
                "resume_from": None,
                "tensorboard_log_dir": str(run_dir / "tensorboard"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "watch_run.py", str(run_dir), "--once", "--no_clear"],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "matrix_lr" in proc.stdout
    assert "demo" in proc.stdout


def test_watch_run_tui_lines_include_recent_events_and_tokens_per_second(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    (tmp_path / "results.tsv").write_text(
        "\t".join(["run_id", "val_bpb", "artifact_bytes", "training_seconds"]) + "\n"
        + "\t".join(["baseline", "1.950000", "5200000", "495.0"]) + "\n"
        + "\t".join(["candidate", "1.870000", "5300000", "496.0"]) + "\n",
        encoding="utf-8",
    )
    events = [
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "run_start", "run_id": "demo", "mode": "train"},
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "train", "step": 3, "train_loss": 4.0, "matrix_lr": 0.01, "step_seconds": 1.0, "total_tokens": 128, "elapsed_training_seconds": 2.0},
        {"schema_version": mod.METRICS_STREAM_VERSION, "event": "val", "phase": "final", "step": 3, "val_loss": 3.5, "val_bpb": 1.9},
    ]
    state = watch_run.collect_run_state(run_dir, metrics_path, run_dir / "results.json", run_dir / "crash.json", events)
    lines = watch_run.render_tui_lines(state, ("train_loss", "tokens_per_second", "val_bpb"), points=40, width=100, height=30)
    text = "\n".join(lines)
    assert "autoresearch-parameter-golf monitor" in text
    assert "Tok/s:" in text
    assert "Recent Run History" in text
    assert "Recent Events" in text
    assert "run_id=candidate" in text
    assert "val/final" in text


def test_watch_run_latest_json_falls_back_to_active_json(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    run_dir = tmp_path / "runs" / "demo"
    index_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    active_json = index_dir / "active.json"
    active_json.write_text(
        json.dumps(
            {
                "status": "running",
                "run_id": "demo",
                "output_dir": str(run_dir),
                "results_path": str(run_dir / "results.json"),
                "metrics_path": str(run_dir / "metrics.jsonl"),
                "tensorboard_log_dir": str(run_dir / "tensorboard"),
                "crash_path": str(run_dir / "crash.json"),
            }
        ),
        encoding="utf-8",
    )
    resolved_run_dir, metrics_path, results_path, crash_path = watch_run.resolve_run_paths(str(index_dir / "latest.json"))
    assert resolved_run_dir == run_dir
    assert metrics_path == run_dir / "metrics.jsonl"
    assert results_path == run_dir / "results.json"
    assert crash_path == run_dir / "crash.json"


def test_compare_runs_render_rows() -> None:
    rows = [
        {"run_id": "run_a", "val_bpb": 1.8, "artifact_bytes": 5_000_000, "training_seconds": 300.0, "num_steps": 100, "depth": 10},
        {"run_id": "run_b", "val_bpb": 1.7, "artifact_bytes": 6_000_000, "training_seconds": 320.0, "num_steps": 120, "depth": 12},
    ]
    text = compare_runs.render_rows(rows, sort_by="val_bpb", limit=10)
    assert "run_b" in text
    assert "1.700000" in text
    assert "artifact_MB" in text


def test_autoresearch_index_script_updates_latest_and_best(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"

    def write_results(run_name: str, val_bpb: float, *, status: str = "success") -> Path:
        results_path = tmp_path / run_name / "results.json"
        return write_fake_results(results_path, run_id=run_name, val_bpb=val_bpb, status=status)

    first = write_results("run_a", 1.50)
    second = write_results("run_b", 1.40)
    subprocess.run(
        [sys.executable, "scripts/index_autoresearch_run.py", str(first), "--index_dir", str(index_dir)],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "scripts/index_autoresearch_run.py", str(second), "--index_dir", str(index_dir)],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    latest = json.loads((index_dir / "latest.json").read_text(encoding="utf-8"))
    best = json.loads((index_dir / "best.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "run_b"
    assert best["run_id"] == "run_b"
    assert float(best["val_bpb"]) == 1.40


def test_autoresearch_state_init_start_finish_and_decide(tmp_path: Path) -> None:
    state_dir = tmp_path / ".autoresearch"
    baseline_results = write_fake_results(tmp_path / "baseline" / "results.json", run_id="baseline", val_bpb=1.80)
    session = autoresearch_state.init_session(state_dir, baseline_results)
    assert session["status"] == "ready"
    assert session["accepted_run_id"] == "baseline"
    assert (state_dir / "notes.md").is_file()

    started = autoresearch_state.start_run(
        state_dir,
        run_id="trial_a",
        output_dir=str(tmp_path / "trial_a"),
        results_path_value=str(tmp_path / "trial_a" / "results.json"),
        metrics_path=str(tmp_path / "trial_a" / "metrics.jsonl"),
        tensorboard_log_dir=str(tmp_path / "trial_a" / "tensorboard"),
        crash_path=str(tmp_path / "trial_a" / "crash.json"),
    )
    assert started["status"] == "running"
    assert started["current_experiment"]["run_id"] == "trial_a"

    trial_results = write_fake_results(tmp_path / "trial_a" / "results.json", run_id="trial_a", val_bpb=1.75)
    finished = autoresearch_state.finish_run(state_dir, trial_results)
    assert finished["status"] == "ready"
    assert finished["latest_run_id"] == "trial_a"
    assert finished["latest_val_bpb"] == 1.75

    decided = autoresearch_state.decide_run(state_dir, "trial_a", "accepted", trial_results)
    assert decided["accepted_run_id"] == "trial_a"
    assert decided["accepted_val_bpb"] == 1.75

    events = [
        json.loads(line)
        for line in (state_dir / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event"] == "baseline_init" for event in events)
    assert any(event["event"] == "run_start" and event["run_id"] == "trial_a" for event in events)
    assert any(event["event"] == "run_result" and event["run_id"] == "trial_a" for event in events)
    assert any(event["event"] == "decision" and event["decision"] == "accepted" for event in events)


def test_run_autoresearch_experiment_requires_initialized_session(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["AUTORESEARCH_ROOT"] = str(tmp_path / "runs")
    env["STATE_DIR"] = str(tmp_path / ".autoresearch")
    env["RUN_ID"] = "ar5090-test"
    proc = subprocess.run(
        ["bash", "scripts/run_autoresearch_experiment.sh"],
        cwd=Path(__file__).parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "No autoresearch session found" in proc.stderr


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_smoke_train_and_export(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.iterations = 2
    summary = mod.train_one_run(cfg)
    assert summary.export is not None
    assert summary.peak_vram_mb >= 0.0


@pytest.mark.ddp
@pytest.mark.skipif(importlib.util.find_spec("torch.distributed.run") is None, reason="torch.distributed.run unavailable")
def test_ddp_smoke_subprocess(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.output_dir = str(tmp_path / "ddp_out")
    cfg.results_tsv_path = str(tmp_path / "ddp_results.tsv")
    cfg.iterations = 2
    cfg.log_every = 1
    cfg.val_every = 1
    cfg.save_final_quantized = False
    cfg_path = tmp_path / "ddp_cfg.json"
    cfg_path.write_text(json.dumps(mod.config_to_dict(cfg), indent=2), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "train.py",
            "--config_json",
            str(cfg_path),
        ],
        cwd=Path(__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "val_bpb:" in proc.stdout
    results = mod.load_and_validate_results(Path(cfg.output_dir) / "results.json")
    assert results["num_steps"] >= 1


@pytest.mark.compile_smoke
@pytest.mark.skipif(not hasattr(torch, "compile") or not torch.cuda.is_available(), reason="compile smoke requires CUDA compile")
def test_torch_compile_smoke(tmp_path: Path) -> None:
    make_easy_shards(tmp_path)
    cfg = tiny_cfg(tmp_path)
    cfg.iterations = 2
    cfg.use_compile = True
    summary = mod.train_one_run(cfg)
    assert summary.num_steps >= 1


@pytest.mark.sentencepiece_real
def test_real_sentencepiece_tokenizer_path_if_available() -> None:
    model_path = os.environ.get("PGOLF_SENTENCEPIECE_MODEL")
    if not model_path:
        pytest.skip("PGOLF_SENTENCEPIECE_MODEL not set")
    path = Path(model_path)
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    tokenizer = mod.load_sentencepiece_model(str(path))
    vocab_size = int(tokenizer.vocab_size())
    base, has_space, is_boundary = mod.build_piece_byte_luts(tokenizer, vocab_size=vocab_size, device=torch.device("cpu"))
    assert base.numel() >= vocab_size
    assert has_space.numel() >= vocab_size
    assert is_boundary.numel() >= vocab_size
