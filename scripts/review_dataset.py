"""CyberCodeMini Review Queue Inspector CLI (Phase 7)

Inspects and manages examples flagged for human review in data/metadata/review_queue.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def inspect_review_queue(queue_path: Path):
    if not queue_path.exists():
        print("Review queue is empty (data/metadata/review_queue.jsonl does not exist).")
        return

    items = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                items.append((line_num, json.loads(line)))

    print("\n" + "=" * 60)
    print(f"  CYBERCODEMINI HUMAN REVIEW QUEUE ({len(items)} items)")
    print("=" * 60)

    for line_num, record in items:
        source_id = record.get("source_id", "unknown")
        category = record.get("category", "unknown")
        reasons = record.get("review_reasons", [])
        print(f"\nItem #{line_num} [ID: {source_id}] [Category: {category}]")
        print(f"  Flagged Reasons: {', '.join(reasons)}")
        ex = record.get("example", {})
        messages = ex.get("messages", [])
        if messages:
            first_user = next((m["content"] for m in messages if m.get("role") == "user"), "N/A")
            print(f"  Prompt Preview:  {first_user[:100]}...")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect human review queue.")
    parser.add_argument("--queue", "-q", default="data/metadata/review_queue.jsonl", help="Review queue path")
    args = parser.parse_args()

    inspect_review_queue(Path(args.queue))
    return 0


if __name__ == "__main__":
    sys.exit(main())
