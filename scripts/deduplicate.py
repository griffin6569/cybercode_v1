#!/usr/bin/env python3
"""Deduplicate a CyberCodeMini JSONL dataset.

Supports:
- Exact duplicate detection (content hash)
- Normalized-text duplicate detection (lowercased, whitespace-collapsed)

Architecture allows embedding-based dedup to be added later.

Usage:
    python scripts/deduplicate.py data/raw/dev_dataset.jsonl -o data/interim/deduped.jsonl
    python scripts/deduplicate.py data/raw/dev_dataset.jsonl --method normalized
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _exact_hash(data: dict) -> str:
    """Hash based on exact message content."""
    messages = data.get("messages", [])
    parts = [f"{m.get('role', '')}:{m.get('content', '')}" for m in messages]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _normalized_hash(data: dict) -> str:
    """Hash based on normalized (lowercased, whitespace-collapsed) content."""
    messages = data.get("messages", [])
    parts = []
    for m in messages:
        content = m.get("content", "").lower()
        content = re.sub(r"\s+", " ", content).strip()
        parts.append(f"{m.get('role', '')}:{content}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


HASH_METHODS = {
    "exact": _exact_hash,
    "normalized": _normalized_hash,
}


def deduplicate(
    input_path: Path,
    output_path: Path,
    method: str = "exact",
) -> dict:
    """Deduplicate a JSONL file.

    Args:
        input_path: Path to input JSONL.
        output_path: Path to write deduplicated output.
        method: Dedup method ("exact" or "normalized").

    Returns:
        Statistics dict with counts.
    """
    hash_fn = HASH_METHODS.get(method)
    if hash_fn is None:
        raise ValueError(f"Unknown dedup method: {method}. Choose from: {list(HASH_METHODS)}")

    seen: set[str] = set()
    total = 0
    kept = 0
    duplicates = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

            h = hash_fn(data)
            if h in seen:
                duplicates += 1
            else:
                seen.add(h)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "method": method,
        "total_examples": total,
        "kept": kept,
        "duplicates_removed": duplicates,
    }
    return stats


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deduplicate a CyberCodeMini JSONL dataset")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--method",
        type=str,
        default="exact",
        choices=list(HASH_METHODS),
        help="Deduplication method (default: exact)",
    )
    parser.add_argument("--json", action="store_true", help="Output stats as JSON")
    parsed = parser.parse_args(args)

    stats = deduplicate(
        Path(parsed.input),
        Path(parsed.output),
        method=parsed.method,
    )

    if parsed.json:
        print(json.dumps(stats, indent=2))
    else:
        print()
        print("=" * 50)
        print("  DEDUPLICATION REPORT")
        print("=" * 50)
        print(f"  Input:      {stats['input_file']}")
        print(f"  Output:     {stats['output_file']}")
        print(f"  Method:     {stats['method']}")
        print(f"  Total:      {stats['total_examples']}")
        print(f"  Kept:       {stats['kept']}")
        print(f"  Removed:    {stats['duplicates_removed']}")
        print("=" * 50)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
