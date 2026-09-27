"""CyberCodeMini Phase 11 Capability & Source Balance Analysis Script

Analyzes the expanded Corpus v0.3.0 capability coverage, dataset source distribution,
license distribution, security distribution, and difficulty breakdown.
Checks for monoculture warnings (> 40% from a single source).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    base_dir = Path(__file__).resolve().parent.parent
    train_file = base_dir / "data" / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.3.0.jsonl"
    val_file = base_dir / "data" / "processed" / "validation_candidates" / "cybercodemini_validation_candidates_v0.3.0.jsonl"

    if not train_file.exists():
        print(f"Error: Expanded dataset '{train_file}' not found.")
        sys.exit(1)

    train_items = [json.loads(line.strip()) for line in open(train_file, encoding="utf-8") if line.strip()]
    val_items = [json.loads(line.strip()) for line in open(val_file, encoding="utf-8") if line.strip()] if val_file.exists() else []

    total_train = len(train_items)

    cap_counts: dict[str, int] = {}
    src_counts: dict[str, int] = {}
    lic_counts: dict[str, int] = {}
    sec_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}

    for item in train_items:
        meta = item.get("metadata", {})
        cap = meta.get("primary_capability") or meta.get("category", "unknown")
        cap_counts[cap] = cap_counts.get(cap, 0) + 1

        src = meta.get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

        lic = meta.get("license", "unknown")
        lic_counts[lic] = lic_counts.get(lic, 0) + 1

        auth = meta.get("authorization", "not_applicable")
        sec_counts[auth] = sec_counts.get(auth, 0) + 1

        diff = meta.get("difficulty", "medium")
        diff_counts[diff] = diff_counts.get(diff, 0) + 1

    print("=" * 68)
    print("CyberCodeMini Phase 11 Balance Analysis (Corpus v0.3.0)")
    print("=" * 68)
    print(f"Total Training Candidates:   {total_train}")
    print(f"Total Validation Candidates: {len(val_items)}\n")

    print(f"{'Capability':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 68)
    for cap, cnt in sorted(cap_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_train) * 100.0 if total_train > 0 else 0
        print(f"{cap:<30} {cnt:<10} {pct:.1f}%")

    print("\n" + "-" * 68)
    print("Source Distribution")
    print("-" * 68)
    monoculture_warning = False
    for src, cnt in sorted(src_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_train) * 100.0 if total_train > 0 else 0
        warn_flag = " [WARNING: >40%]" if pct > 40.0 else ""
        if pct > 40.0:
            monoculture_warning = True
        print(f"{src:<42} {cnt:<6} {pct:.1f}%{warn_flag}")

    print("\n" + "-" * 68)
    print("License Distribution")
    print("-" * 68)
    for lic, cnt in sorted(lic_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_train) * 100.0 if total_train > 0 else 0
        print(f"{lic:<30} {cnt:<10} {pct:.1f}%")

    print("\n" + "-" * 68)
    print("Security Authorization Distribution")
    print("-" * 68)
    for auth, cnt in sorted(sec_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_train) * 100.0 if total_train > 0 else 0
        print(f"{auth:<30} {cnt:<10} {pct:.1f}%")

    print("\n" + "-" * 68)
    print("Difficulty Distribution")
    print("-" * 68)
    for diff, cnt in sorted(diff_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_train) * 100.0 if total_train > 0 else 0
        print(f"{diff:<30} {cnt:<10} {pct:.1f}%")

    print("=" * 68)
    if monoculture_warning:
        print("Note: Monoculture warning flagged for high single-source concentration.")
    else:
        print("Dataset diversity check PASSED: No single source exceeds 40%.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
