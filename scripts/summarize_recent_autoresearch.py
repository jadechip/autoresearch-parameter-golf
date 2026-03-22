from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


KEYWORD_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    (
        "xsa_or_cache",
        (
            "xsa",
            "cache",
            "kv cache",
            "cross-window",
            "cross window",
            "sliding-window",
            "sliding window",
            "partial rope",
            "partial-rope",
        ),
    ),
    (
        "checkpoint_soup_or_ptq",
        (
            "soup",
            "ptq",
            "post-quant",
            "post quant",
            "checkpoint",
            "swa",
            "warmdown snapshot",
            "snapshot",
        ),
    ),
    (
        "smeargate_or_ttt",
        (
            "smear",
            "ttt",
            "meta-ttt",
            "meta ttt",
            "causal ttt",
            "causal-ttt",
        ),
    ),
    (
        "canon_or_neighbor_mixer",
        (
            "canon",
            "acd",
            "neighbor",
            "neighbour",
            "neighboring-token",
            "neighbouring-token",
        ),
    ),
    (
        "selective_precision_or_quantization",
        (
            "float",
            "fake quant",
            "fake-quant",
            "qat",
            "precision",
            "proj",
            "q_proj",
            "k/v",
            "out_proj",
            "fc",
        ),
    ),
    (
        "low_rank_q_reallocation",
        (
            "low-rank q",
            "low rank q",
            "shared-q",
            "shared q",
            "q-rank",
            "q rank",
            "full-rank q",
            "full rank q",
        ),
    ),
    (
        "batch_or_context_curriculum",
        (
            "batch",
            "seq",
            "context",
            "curriculum",
            "warmup",
            "single-step",
            "single step",
            "mid-batch",
            "mid floor",
        ),
    ),
    (
        "local_token_module",
        (
            "mixer",
            "bigram",
            "local",
            "token module",
        ),
    ),
    (
        "carrier_repartition",
        (
            "carrier",
            "tail",
            "shared",
            "depth",
            "repartition",
            "reallocate",
            "unique",
            "loop",
            "mlp",
            "d_model",
        ),
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_log_messages(limit: int) -> list[str]:
    proc = subprocess.run(
        ["git", "log", "--oneline", f"--max-count={limit}", "--grep=^autoresearch:"],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def classify_commit(message: str) -> str:
    lower = message.lower()
    for family, keywords in KEYWORD_FAMILIES:
        if any(keyword in lower for keyword in keywords):
            return family
    return "other"


def summarize(limit: int) -> dict[str, object]:
    messages = git_log_messages(limit)
    families = [classify_commit(message) for message in messages]
    counts = Counter(families)
    dominant_family, dominant_count = counts.most_common(1)[0] if counts else ("none", 0)
    recent_same_family_streak = 0
    if families:
        first = families[0]
        for family in families:
            if family != first:
                break
            recent_same_family_streak += 1
    warnings: list[str] = []
    if dominant_count >= max(3, limit // 2):
        warnings.append(f"dominant_recent_family={dominant_family}")
    if recent_same_family_streak >= 3:
        warnings.append(f"same_family_streak={recent_same_family_streak}")
    if families[:4] and len(set(families[:4])) == 1:
        warnings.append(f"last4_same_family={families[0]}")
    return {
        "limit": limit,
        "dominant_family": dominant_family,
        "dominant_count": dominant_count,
        "recent_same_family_streak": recent_same_family_streak,
        "family_counts": dict(counts),
        "warnings": warnings,
        "recent": [
            {"family": family, "message": message}
            for family, message in zip(families, messages, strict=False)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recent autoresearch experiment families from git history.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = summarize(args.limit)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"recent_autoresearch_commits={payload['limit']}")
    print(f"dominant_family={payload['dominant_family']} count={payload['dominant_count']}")
    print(f"recent_same_family_streak={payload['recent_same_family_streak']}")
    counts = payload["family_counts"]
    print("family_counts:")
    for family, count in sorted(counts.items()):
        print(f"  {family}: {count}")
    warnings = payload["warnings"]
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  {warning}")
    print("recent:")
    for item in payload["recent"]:
        print(f"  [{item['family']}] {item['message']}")


if __name__ == "__main__":
    main()
