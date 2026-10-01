"""Prepare validated, model-independent Open-Jev JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_NAME = "ZefanCai/Open-Jev"
DATASET_CONFIG = "release-v2-redistributable"
SCHEMA_VERSION = "decision-jsonl-v1"


def normalized_key(row: dict[str, Any]) -> str:
    def clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    payload = [clean(row["state"]), clean(row["question"]), [clean(x["text"]) for x in row["options"]]]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def canonicalize(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw_state = json.loads(raw["state_json"])
        state = raw_state if isinstance(raw_state, str) else json.dumps(raw_state, ensure_ascii=False, sort_keys=True)
        raw_options = raw["options"]
        target = [float(x) for x in raw["target"]]
        if len(raw_options) != len(target) or not raw_options:
            return None, "length_mismatch"
        if any(not math.isfinite(x) or x < 0 for x in target):
            return None, "invalid_target"
        total = sum(target)
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            return None, "target_not_distribution"
        if total != 1.0:
            target = [x / total for x in target]
        options = []
        for index, option in enumerate(raw_options):
            if isinstance(option, dict):
                text = str(option.get("text", option.get("name", "")))
                option_id = str(option.get("id") or f"{raw['id']}:option:{index}")
            else:
                text = str(option)
                option_id = f"{raw['id']}:option:{index}"
            options.append({"id": option_id, "text": text})
        if len({x["id"] for x in options}) != len(options):
            return None, "duplicate_candidate_id"
        return {
            "id": str(raw["id"]),
            "source": str(raw.get("source") or DATASET_NAME),
            "group_id": str(raw.get("group_id") or raw["id"]),
            "state": state,
            "question": str(raw["question"]),
            "options": options,
            "target": target,
        }, None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, "malformed"


def deterministic_sample(rows: list[dict[str, Any]], size: int, seed: int, split: str) -> list[dict[str, Any]]:
    if size < 0 or size >= len(rows):
        return rows
    rng = random.Random(f"{seed}:{split}")
    indices = sorted(rng.sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def summarize(splits: dict[str, list[dict[str, Any]]], rejected: Counter) -> None:
    for name, rows in splits.items():
        option_counts = Counter(len(row["options"]) for row in rows)
        entropy = [
            -sum(p * math.log(p) for p in row["target"] if p > 0)
            for row in rows
        ]
        mean_entropy = sum(entropy) / len(entropy) if entropy else 0.0
        print(f"{name}: {len(rows)} rows; options={dict(sorted(option_counts.items()))}; mean_target_entropy={mean_entropy:.6f}")
    print(f"rejected: {dict(sorted(rejected.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/poc")
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream source Parquet shards instead of materializing an Arrow cache",
    )
    parser.add_argument(
        "--cache-dir",
        help="optional Hugging Face cache location on a filesystem with sufficient space",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET_NAME).sha
    load_kwargs = {
        "revision": revision,
        "streaming": args.streaming,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, **load_kwargs)
    official_order = [name for name in ("train", "calibration", "validation", "test", "ood") if name in dataset]
    rejected: Counter = Counter()
    canonical: dict[str, list[dict[str, Any]]] = {}
    for split in official_order:
        canonical[split] = []
        for raw in dataset[split]:
            row, reason = canonicalize(raw)
            if row is None:
                rejected[reason or "unknown"] += 1
            else:
                canonical[split].append(row)

    # Assign duplicates to the most held-out split. This is done before sampling.
    seen: set[str] = set()
    deduplicated: dict[str, list[dict[str, Any]]] = {}
    for split in reversed(official_order):
        kept = []
        for row in canonical[split]:
            key = hashlib.sha256(normalized_key(row).encode()).hexdigest()
            if key in seen:
                rejected[f"cross_split_duplicate_from_{split}"] += 1
            else:
                seen.add(key)
                kept.append(row)
        deduplicated[split] = kept

    requested = {"train": args.train_size, "validation": args.validation_size, "test": args.test_size}
    final = {
        split: deterministic_sample(rows, requested.get(split, -1), args.seed, split)
        for split, rows in deduplicated.items()
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in final.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": DATASET_NAME,
        "configuration": DATASET_CONFIG,
        "revision": revision,
        "creation_arguments": vars(args),
        "counts": {name: len(rows) for name, rows in final.items()},
        "rejected": dict(rejected),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summarize(final, rejected)


if __name__ == "__main__":
    main()
