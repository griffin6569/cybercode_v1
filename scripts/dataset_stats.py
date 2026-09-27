#!/usr/bin/env python3
"""Compute statistics for a CyberCodeMini JSONL dataset.

Reports:
- Total examples, categories, languages, difficulty distribution
- Source datasets, synthetic percentage, tool-use percentage
- Cybersecurity category breakdown
- Train/validation/test sizes (if split files exist)

Usage:
    python scripts/dataset_stats.py data/raw/dev_dataset.jsonl
    python scripts/dataset_stats.py data/raw/dev_dataset.jsonl --json
    python scripts/dataset_stats.py data/ --splits   # Report on all splits
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_stats(file_path: Path) -> dict:
    """Compute statistics for a single JSONL file."""
    total = 0
    categories: Counter = Counter()
    languages: Counter = Counter()
    difficulties: Counter = Counter()
    sources: Counter = Counter()
    environments: Counter = Counter()
    vulnerability_types: Counter = Counter()
    synthetic_count = 0
    tool_use_count = 0
    authorization_counts: Counter = Counter()
    total_messages = 0
    total_chars = 0
    min_messages = float("inf")
    max_messages = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            messages = data.get("messages", [])
            meta = data.get("metadata", {})

            # Message stats
            n_msgs = len(messages)
            total_messages += n_msgs
            min_messages = min(min_messages, n_msgs)
            max_messages = max(max_messages, n_msgs)
            total_chars += sum(len(m.get("content", "")) for m in messages)

            # Category
            cat = meta.get("category", "unknown")
            categories[cat] += 1

            # Language
            lang = meta.get("language")
            if lang:
                languages[lang] += 1

            # Difficulty
            diff = meta.get("difficulty")
            if diff:
                difficulties[diff] += 1

            # Source
            source = meta.get("source", "unknown")
            sources[source] += 1

            # Environment
            env = meta.get("environment")
            if env:
                environments[env] += 1

            # Vulnerability type
            vuln = meta.get("vulnerability_type")
            if vuln:
                vulnerability_types[vuln] += 1

            # Synthetic
            if meta.get("synthetic", False):
                synthetic_count += 1

            # Authorization
            auth = meta.get("authorization")
            if auth:
                authorization_counts[auth] += 1

            # Tool use (check if any message has tool role or tool_calls)
            has_tools = any(
                m.get("role") in ("tool", "tool_result") or m.get("tool_calls")
                for m in messages
            )
            if has_tools:
                tool_use_count += 1

    stats = {
        "file": str(file_path),
        "total_examples": total,
        "total_messages": total_messages,
        "total_characters": total_chars,
        "avg_messages_per_example": round(total_messages / total, 2) if total else 0,
        "min_messages": min_messages if min_messages != float("inf") else 0,
        "max_messages": max_messages,
        "categories": dict(categories.most_common()),
        "languages": dict(languages.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "sources": dict(sources.most_common()),
        "environments": dict(environments.most_common()),
        "vulnerability_types": dict(vulnerability_types.most_common()),
        "authorization": dict(authorization_counts.most_common()),
        "synthetic_count": synthetic_count,
        "synthetic_percentage": round(100 * synthetic_count / total, 1) if total else 0,
        "tool_use_count": tool_use_count,
        "tool_use_percentage": round(100 * tool_use_count / total, 1) if total else 0,
    }
    return stats


def compute_split_stats(data_dir: Path) -> dict:
    """Compute stats for train/validation/test splits."""
    result = {}
    for split_name in ["train", "validation", "test"]:
        split_dir = data_dir / split_name
        jsonl_files = list(split_dir.glob("*.jsonl"))
        if jsonl_files:
            result[split_name] = compute_stats(jsonl_files[0])
        else:
            result[split_name] = {"total_examples": 0, "note": "No JSONL file found"}
    return result


def format_stats(stats: dict) -> str:
    """Format stats as a human-readable report."""
    lines = [
        "",
        "=" * 60,
        "  DATASET STATISTICS",
        "=" * 60,
        f"  File:                   {stats['file']}",
        f"  Total examples:         {stats['total_examples']}",
        f"  Total messages:         {stats['total_messages']}",
        f"  Total characters:       {stats['total_characters']:,}",
        f"  Avg messages/example:   {stats['avg_messages_per_example']}",
        f"  Min/Max messages:       {stats['min_messages']}/{stats['max_messages']}",
        "",
        "  CATEGORIES:",
    ]
    for cat, count in stats["categories"].items():
        lines.append(f"    {cat:30s} {count:5d}")

    if stats["languages"]:
        lines.append("")
        lines.append("  LANGUAGES:")
        for lang, count in stats["languages"].items():
            lines.append(f"    {lang:30s} {count:5d}")

    if stats["difficulties"]:
        lines.append("")
        lines.append("  DIFFICULTY:")
        for diff, count in stats["difficulties"].items():
            lines.append(f"    {diff:30s} {count:5d}")

    if stats["vulnerability_types"]:
        lines.append("")
        lines.append("  VULNERABILITY TYPES:")
        for vt, count in stats["vulnerability_types"].items():
            lines.append(f"    {vt:30s} {count:5d}")

    if stats["sources"]:
        lines.append("")
        lines.append("  SOURCES:")
        for src, count in stats["sources"].items():
            lines.append(f"    {src:30s} {count:5d}")

    lines.extend([
        "",
        f"  Synthetic:              {stats['synthetic_count']} ({stats['synthetic_percentage']}%)",
        f"  Tool-use examples:      {stats['tool_use_count']} ({stats['tool_use_percentage']}%)",
        "=" * 60,
        "",
    ])
    return "\n".join(lines)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute CyberCodeMini dataset statistics")
    parser.add_argument("path", type=str, help="JSONL file or data directory")
    parser.add_argument("--splits", action="store_true", help="Report on train/val/test splits")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parsed = parser.parse_args(args)

    path = Path(parsed.path)

    if parsed.splits and path.is_dir():
        all_stats = compute_split_stats(path)
        if parsed.json:
            print(json.dumps(all_stats, indent=2))
        else:
            for split_name, stats in all_stats.items():
                if stats.get("total_examples", 0) > 0:
                    print(f"\n  --- {split_name.upper()} SPLIT ---")
                    print(format_stats(stats))
                else:
                    print(f"\n  --- {split_name.upper()} SPLIT ---")
                    print(f"  No data found.\n")
    else:
        if path.is_dir():
            # Find first JSONL file
            jsonl_files = list(path.rglob("*.jsonl"))
            if not jsonl_files:
                print(f"No JSONL files found in {path}")
                return 1
            path = jsonl_files[0]

        stats = compute_stats(path)
        if parsed.json:
            print(json.dumps(stats, indent=2))
        else:
            print(format_stats(stats))

    return 0


if __name__ == "__main__":
    sys.exit(main())
