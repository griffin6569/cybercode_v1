#!/usr/bin/env python3
"""Inspect individual examples from a CyberCodeMini JSONL dataset.

Usage:
    python scripts/inspect_dataset.py data/raw/dev_dataset.jsonl
    python scripts/inspect_dataset.py data/raw/dev_dataset.jsonl --index 3
    python scripts/inspect_dataset.py data/raw/dev_dataset.jsonl --category security_review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def inspect_example(data: dict, index: int) -> str:
    """Format a single example for display."""
    lines = [
        f"\n{'─' * 60}",
        f"  Example #{index}",
        f"{'─' * 60}",
    ]

    # Metadata
    meta = data.get("metadata", {})
    lines.append(f"  Category:      {meta.get('category', 'N/A')}")
    lines.append(f"  Difficulty:    {meta.get('difficulty', 'N/A')}")
    lines.append(f"  Language:      {meta.get('language', 'N/A')}")
    lines.append(f"  Environment:   {meta.get('environment', 'N/A')}")
    lines.append(f"  Authorization: {meta.get('authorization', 'N/A')}")
    lines.append(f"  Source:        {meta.get('source', 'N/A')}")
    lines.append(f"  Synthetic:     {meta.get('synthetic', 'N/A')}")
    lines.append("")

    # Messages
    messages = data.get("messages", [])
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Truncate long content
        if len(content) > 500:
            content = content[:500] + f"\n... [{len(content) - 500} chars truncated]"
        lines.append(f"  [{role}]")
        for line in content.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect CyberCodeMini dataset examples")
    parser.add_argument("file", type=str, help="JSONL file to inspect")
    parser.add_argument("--index", type=int, default=None, help="Show only example at this index (0-based)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=5, help="Max examples to show")
    parsed = parser.parse_args(args)

    file_path = Path(parsed.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"\nFile: {file_path}")
    print(f"Total examples: {len(examples)}")

    if parsed.index is not None:
        if 0 <= parsed.index < len(examples):
            print(inspect_example(examples[parsed.index], parsed.index))
        else:
            print(f"Index {parsed.index} out of range (0-{len(examples) - 1})")
            return 1
    else:
        shown = 0
        for i, ex in enumerate(examples):
            if parsed.category:
                if ex.get("metadata", {}).get("category") != parsed.category:
                    continue
            print(inspect_example(ex, i))
            shown += 1
            if shown >= parsed.limit:
                remaining = len(examples) - shown
                if remaining > 0:
                    print(f"\n  ... {remaining} more examples (use --limit to see more)")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
