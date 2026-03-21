from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from train import load_and_validate_results, resolve_artifact_dir


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    chars = []
    prev_dash = False
    for ch in lowered:
        if ch.isalnum():
            chars.append(ch)
            prev_dash = False
        elif not prev_dash:
            chars.append("-")
            prev_dash = True
    slug = "".join(chars).strip("-")
    return slug or "submission"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.9f}"


class SubmissionPackagerError(RuntimeError):
    pass


def load_run_context(results_json: Path) -> dict[str, Any]:
    validated = dict(load_and_validate_results(results_json))
    raw = load_json(results_json)
    output_dir = Path(validated["output_dir"])
    artifact_dir = None
    manifest = None
    artifact_payload = raw.get("artifact")
    if artifact_payload is not None:
        artifact_dir = resolve_artifact_dir(artifact_payload["artifact_dir"])
        manifest = load_json(artifact_dir / "manifest.json")
    return {
        "results_json": results_json,
        "results": raw,
        "validated": validated,
        "output_dir": output_dir,
        "artifact_dir": artifact_dir,
        "manifest": manifest,
        "train_log_path": output_dir / "train.log",
        "eval_log_path": output_dir / "eval.log",
        "config_path": output_dir / "config.json",
        "export_stats_path": output_dir / "export_stats.json",
        "artifact_reload_eval_path": output_dir / "artifact_reload_eval.json",
    }


def score_contexts(train_contexts: list[dict[str, Any]], eval_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if eval_contexts:
        return eval_contexts
    return train_contexts


def summary_metrics(contexts: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    losses = [ctx["results"].get("val_loss") for ctx in contexts if ctx["results"].get("val_loss") is not None]
    bpbs = [ctx["results"].get("val_bpb") for ctx in contexts if ctx["results"].get("val_bpb") is not None]
    mean_loss = mean(losses) if losses else None
    mean_bpb = mean(bpbs) if bpbs else None
    return mean_loss, mean_bpb


def artifact_summary(train_contexts: list[dict[str, Any]]) -> dict[str, int]:
    manifests = [ctx["manifest"] for ctx in train_contexts if ctx["manifest"] is not None]
    if not manifests:
        raise SubmissionPackagerError("At least one training results.json with an exported artifact is required.")
    artifact_bytes = [int(manifest["byte_counts"]["artifact_bytes"]) for manifest in manifests]
    code_bytes = [int(manifest["byte_counts"]["code_bytes"]) for manifest in manifests]
    model_bytes = [int(manifest["byte_counts"]["compressed_model_bytes"]) for manifest in manifests]
    return {
        "artifact_bytes": max(artifact_bytes),
        "code_bytes": max(code_bytes),
        "model_bytes": max(model_bytes),
    }


def copy_counted_code(first_train: dict[str, Any], package_dir: Path) -> list[str]:
    manifest = first_train["manifest"]
    artifact_dir = first_train["artifact_dir"]
    if manifest is None or artifact_dir is None:
        raise SubmissionPackagerError("Training run is missing artifact manifest/code bundle.")
    copied: list[str] = []
    counted_files = list(manifest.get("counted_files") or [])
    if not counted_files:
        raise SubmissionPackagerError("Artifact manifest does not contain counted_files entries.")
    for entry in counted_files:
        src = artifact_dir / entry["artifact_relpath"]
        rel = Path(entry["artifact_relpath"])
        if rel.parts[:1] == ("code",):
            rel = Path(*rel.parts[1:])
        if rel == Path("train.py"):
            dest = package_dir / "train_gpt.py"
        else:
            dest = package_dir / rel
        copy_if_exists(src, dest)
        copied.append(str(dest.relative_to(package_dir)))
    return copied


def copy_support_files(context: dict[str, Any], dest_dir: Path) -> None:
    copy_if_exists(context["results_json"], dest_dir / "results.json")
    copy_if_exists(context["config_path"], dest_dir / "config.json")
    copy_if_exists(context["export_stats_path"], dest_dir / "export_stats.json")
    copy_if_exists(context["artifact_reload_eval_path"], dest_dir / "artifact_reload_eval.json")
    if context["manifest"] is not None and context["artifact_dir"] is not None:
        copy_if_exists(context["artifact_dir"] / "manifest.json", dest_dir / "artifact_manifest.json")
    copy_if_exists(context["train_log_path"], dest_dir / "train.log")
    copy_if_exists(context["eval_log_path"], dest_dir / "eval.log")


def build_submission_payload(
    *,
    args: argparse.Namespace,
    train_contexts: list[dict[str, Any]],
    eval_contexts: list[dict[str, Any]],
    package_dir: Path,
) -> dict[str, Any]:
    scored = score_contexts(train_contexts, eval_contexts)
    artifact = artifact_summary(train_contexts)
    mean_loss, mean_bpb = summary_metrics(scored)
    timestamp = args.date or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload: dict[str, Any] = {
        "track": args.track,
        "date": timestamp,
        "name": args.name,
        "author": args.author,
        "github_id": args.github_id,
        "blurb": args.blurb,
        "artifact_bytes": artifact["artifact_bytes"],
        "bytes_total": artifact["artifact_bytes"],
        "code_bytes": artifact["code_bytes"],
        "bytes_code": artifact["code_bytes"],
        "compressed_model_bytes": artifact["model_bytes"],
        "bytes_model_int8_zlib": artifact["model_bytes"],
        "package_dir": str(package_dir),
    }
    if args.hardware:
        payload["hardware"] = args.hardware
    if args.hardware_note:
        payload["hardware_note"] = args.hardware_note
    if args.p_value is not None:
        payload["p_value"] = args.p_value

    if len(scored) == 1:
        result = scored[0]["results"]
        payload["val_loss"] = result.get("val_loss")
        payload["val_bpb"] = result.get("val_bpb")
        payload["run_id"] = result.get("run_id")
    else:
        seed_results: dict[str, Any] = {}
        for ctx in scored:
            result = ctx["results"]
            seed_results[str(result["run_id"])] = {
                "val_loss": result.get("val_loss"),
                "val_bpb": result.get("val_bpb"),
                "steps": result.get("num_steps"),
                "total_seconds": result.get("total_seconds"),
            }
        payload["seed_results"] = seed_results
        payload["mean_val_loss"] = mean_loss
        payload["mean_val_bpb"] = mean_bpb

    return payload


def render_results_rows(contexts: list[dict[str, Any]], *, include_mode: bool) -> list[str]:
    lines = [
        "| run_id | mode | val_loss | val_bpb | total_seconds | artifact_bytes |"
        if include_mode
        else "| run_id | val_loss | val_bpb | total_seconds | artifact_bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: |"
        if include_mode
        else "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ctx in contexts:
        result = ctx["results"]
        mode = result.get("mode", "train")
        row = [
            str(result["run_id"]),
            format_float(result.get("val_loss")),
            format_float(result.get("val_bpb")),
            format_float(result.get("total_seconds")),
            str(result.get("artifact_bytes", "n/a")),
        ]
        if include_mode:
            lines.append(f"| {row[0]} | {mode} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
        else:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    return lines


def render_readme(
    *,
    args: argparse.Namespace,
    package_dir: Path,
    submission_payload: dict[str, Any],
    train_contexts: list[dict[str, Any]],
    eval_contexts: list[dict[str, Any]],
    copied_code_files: list[str],
) -> str:
    heading = "Record Candidate" if args.track == "track_10min_16mb" else "Non-Record Candidate"
    lines = [
        f"# {heading}: {args.name}",
        "",
        f"**Track**: `{args.track}`",
        f"**Author**: {args.author} (`{args.github_id}`)",
        f"**Date**: {submission_payload['date']}",
        "",
        "## Summary",
        "",
        args.blurb,
        "",
        "## Training Runs",
        "",
        *render_results_rows(train_contexts, include_mode=False),
    ]
    if eval_contexts:
        lines.extend(
            [
                "",
                "## Evaluation Runs",
                "",
                *render_results_rows(eval_contexts, include_mode=False),
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact Summary",
            "",
            f"- `artifact_bytes`: {submission_payload['artifact_bytes']}",
            f"- `code_bytes`: {submission_payload['code_bytes']}",
            f"- `compressed_model_bytes`: {submission_payload['compressed_model_bytes']}",
        ]
    )
    if args.hardware:
        lines.extend(["", "## Hardware", "", f"- `{args.hardware}`"])
        if args.hardware_note:
            lines.append(f"- {args.hardware_note}")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "The packaged training entrypoint is `train_gpt.py`, matching the official record-folder expectation.",
            "",
            "Example training command:",
            "",
            "```bash",
            f"RUN_ID={slugify(args.name)} \\",
            f"torchrun --standalone --nproc_per_node={args.reproduction_nproc} train_gpt.py",
            "```",
            "",
            "Example evaluation command against a saved artifact bundle:",
            "",
            "```bash",
            f"torchrun --standalone --nproc_per_node={args.reproduction_nproc} train_gpt.py \\",
            "  --evaluate_only \\",
            "  --load_artifact_path <path-to-submission_bundle> \\",
            "  --val_pattern ./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin \\",
            "  --tokenizer_path ./data/tokenizers/fineweb_1024_bpe.model",
            "```",
            "",
            "## Included Files",
            "",
            *[f"- `{path}`" for path in sorted(copied_code_files)],
            "- `submission.json`",
            "- `train_runs/` with copied results, configs, manifests, and logs",
        ]
    )
    if eval_contexts:
        lines.append("- `eval_runs/` with copied eval results and logs")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This package was generated from the local experiment repo with `pgolf-package-submission`.",
            "- OpenAI requires the record folder to be self-contained and for all counted code to live in `train_gpt.py`.",
            f"- Generated at `{package_dir}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package one or more runs into a Parameter Golf records-folder candidate.")
    parser.add_argument("--train_results_json", type=str, nargs="+", required=True, help="One or more successful training results.json files.")
    parser.add_argument("--eval_results_json", type=str, nargs="*", default=(), help="Optional evaluate_only results.json files to use for final scoring metadata.")
    parser.add_argument("--track", type=str, choices=("track_10min_16mb", "track_non_record_16mb"), required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--author", type=str, required=True)
    parser.add_argument("--github_id", type=str, required=True)
    parser.add_argument("--blurb", type=str, required=True)
    parser.add_argument("--slug", type=str, default=None, help="Optional folder slug. Defaults to a slugified version of --name.")
    parser.add_argument("--date", type=str, default=None, help="Optional timestamp/date string for submission.json.")
    parser.add_argument("--hardware", type=str, default=None)
    parser.add_argument("--hardware_note", type=str, default=None)
    parser.add_argument("--p_value", type=float, default=None)
    parser.add_argument("--output_root", type=str, default="./submission_candidates")
    parser.add_argument("--reproduction_nproc", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_contexts = [load_run_context(Path(path)) for path in args.train_results_json]
    eval_contexts = [load_run_context(Path(path)) for path in args.eval_results_json]
    slug = args.slug or slugify(args.name)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d") if args.date is None else args.date[:10]
    package_dir = Path(args.output_root) / args.track / f"{stamp}_{slug}"
    if package_dir.exists():
        raise SystemExit(f"Package directory already exists: {package_dir}")
    package_dir.mkdir(parents=True, exist_ok=False)

    copied_code_files = copy_counted_code(train_contexts[0], package_dir)

    train_runs_dir = package_dir / "train_runs"
    for ctx in train_contexts:
        dest = train_runs_dir / str(ctx["results"]["run_id"])
        dest.mkdir(parents=True, exist_ok=True)
        copy_support_files(ctx, dest)

    if eval_contexts:
        eval_runs_dir = package_dir / "eval_runs"
        for ctx in eval_contexts:
            dest = eval_runs_dir / str(ctx["results"]["run_id"])
            dest.mkdir(parents=True, exist_ok=True)
            copy_support_files(ctx, dest)

    submission_payload = build_submission_payload(
        args=args,
        train_contexts=train_contexts,
        eval_contexts=eval_contexts,
        package_dir=package_dir,
    )
    (package_dir / "submission.json").write_text(json.dumps(submission_payload, indent=2) + "\n", encoding="utf-8")
    (package_dir / "README.md").write_text(
        render_readme(
            args=args,
            package_dir=package_dir,
            submission_payload=submission_payload,
            train_contexts=train_contexts,
            eval_contexts=eval_contexts,
            copied_code_files=copied_code_files,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "packaged": True,
                "package_dir": str(package_dir),
                "submission_json": str(package_dir / "submission.json"),
                "readme": str(package_dir / "README.md"),
                "train_runs": [str(train_runs_dir / str(ctx["results"]["run_id"])) for ctx in train_contexts],
                "eval_runs": [
                    str((package_dir / "eval_runs") / str(ctx["results"]["run_id"]))
                    for ctx in eval_contexts
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
