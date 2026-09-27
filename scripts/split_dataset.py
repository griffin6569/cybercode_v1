#!/usr/bin/env python3
"""Split a CyberCodeMini JSONL dataset into train/validation/test sets.

Supports:
- Random splitting
- Group-based splitting (by metadata field) to prevent data leakage

Usage:
    python scripts/split_dataset.py data/interim/deduped.jsonl --output-dir data/
    python scripts/split_dataset.py data/interim/deduped.jsonl --strategy group --group-key source
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_examples(path: Path) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def _write_examples(examples: list[dict], path: Path) -> None:
    """Write examples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def split_random(
    examples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Randomly split examples into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def split_by_group(
    examples: list[dict],
    group_key: str = "source",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by group to prevent data leakage.

    Groups examples by a metadata field, then assigns entire groups
    to train/val/test rather than individual examples.
    """
    rng = random.Random(seed)

    # Group examples
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        key = ex.get("metadata", {}).get(group_key, "__unknown__")
        groups[str(key)].append(ex)

    # Shuffle group keys
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    # Assign groups to splits based on cumulative example count
    total = len(examples)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    train, val, test = [], [], []
    train_count = 0
    val_count = 0

    for key in group_keys:
        group_examples = groups[key]
        if train_count < train_target:
            train.extend(group_examples)
            train_count += len(group_examples)
        elif val_count < val_target:
            val.extend(group_examples)
            val_count += len(group_examples)
        else:
            test.extend(group_examples)

    # Ensure no split is empty if possible
    if not val and test:
        val.append(test.pop(0))
    if not test and len(train) > 2:
        test.append(train.pop())

    return train, val, test


def split_dataset(
    input_path: Path,
    output_dir: Path,
    strategy: str = "random",
    group_key: str = "source",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split a dataset and write the splits to disk.

    Returns statistics dict.
    """
    examples = _load_examples(input_path)

    if not examples:
        raise ValueError(f"No examples found in {input_path}")

    if strategy == "group":
        train, val, test = split_by_group(
            examples, group_key=group_key,
            train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
        )
    else:
        train, val, test = split_random(
            examples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
        )

    # Write splits
    train_path = output_dir / "train" / "train.jsonl"
    val_path = output_dir / "validation" / "validation.jsonl"
    test_path = output_dir / "test" / "test.jsonl"

    _write_examples(train, train_path)
    _write_examples(val, val_path)
    _write_examples(test, test_path)

    stats = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "strategy": strategy,
        "seed": seed,
        "total": len(examples),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "train_ratio_actual": round(len(train) / len(examples), 3) if examples else 0,
        "val_ratio_actual": round(len(val) / len(examples), 3) if examples else 0,
        "test_ratio_actual": round(len(test) / len(examples), 3) if examples else 0,
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "test_file": str(test_path),
    }
    return stats


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a CyberCodeMini dataset")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--strategy", choices=["random", "group"], default="random",
        help="Split strategy",
    )
    parser.add_argument("--group-key", type=str, default="source", help="Metadata key for group split")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Output stats as JSON")
    parsed = parser.parse_args(args)

    stats = split_dataset(
        Path(parsed.input),
        Path(parsed.output_dir),
        strategy=parsed.strategy,
        group_key=parsed.group_key,
        train_ratio=parsed.train_ratio,
        val_ratio=parsed.val_ratio,
        seed=parsed.seed,
    )

    if parsed.json:
        print(json.dumps(stats, indent=2))
    else:
        print()
        print("=" * 50)
        print("  DATASET SPLIT REPORT")
        print("=" * 50)
        print(f"  Strategy:    {stats['strategy']}")
        print(f"  Seed:        {stats['seed']}")
        print(f"  Total:       {stats['total']}")
        print(f"  Train:       {stats['train']} ({stats['train_ratio_actual']:.1%})")
        print(f"  Validation:  {stats['validation']} ({stats['val_ratio_actual']:.1%})")
        print(f"  Test:        {stats['test']} ({stats['test_ratio_actual']:.1%})")
        print("=" * 50)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
