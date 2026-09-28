"""CyberCodeMini Capability Gap Analysis Script (Phase 11)

Analyzes the current training candidate dataset against the Phase 11 target capability distribution
and reports capability counts, target ranges, gaps, and percentage coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.schemas.capabilities import TARGET_CAPABILITY_RANGES, Capability


def map_category_to_capability(category: str) -> str:
    """Map legacy category string to canonical capability string."""
    cat_lower = str(category).lower()
    mapping = {
        "code_generation": "code_generation",
        "debugging": "debugging",
        "security_review": "security_review",
        "vulnerability_detection": "vulnerability_detection",
        "vulnerability_remediation": "vulnerability_remediation",
        "security_reasoning": "security_reasoning",
        "agent_trajectories": "agent_tool_use",
        "agent_tool_use": "agent_tool_use",
        "authorized_lab": "authorized_lab",
        "ctf_challenge": "authorized_lab",
        "ctf": "authorized_lab",
        "test_generation": "test_generation",
        "software_engineering": "software_engineering",
        "refactoring": "refactoring",
    }
    return mapping.get(cat_lower, "code_generation")


def analyze_gaps(train_file: Path, target_total: int = 1000) -> dict:
    counts: dict[str, int] = {cap.value: 0 for cap in Capability}
    total_examples = 0

    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                meta = item.get("metadata", {})
                cap = meta.get("primary_capability")
                if not cap:
                    cat = meta.get("category", "code_generation")
                    cap = map_category_to_capability(cat)
                counts[cap] = counts.get(cap, 0) + 1
                total_examples += 1

    gap_report = []
    print("=" * 68)
    print("CyberCodeMini Capability Gap Analysis")
    print("=" * 68)
    print(f"Target Total Training Corpus: {target_total} examples")
    print(f"Current Training Candidates: {total_examples} examples\n")
    print(f"{'Capability':<26} {'Current':<9} {'Target Range':<15} {'Target %':<10} {'Gap':<6}")
    print("-" * 68)

    for cap_name, (min_pct, max_pct) in TARGET_CAPABILITY_RANGES.items():
        curr_cnt = counts.get(cap_name, 0)
        min_target = int(target_total * (min_pct / 100.0))
        max_target = int(target_total * (max_pct / 100.0))
        target_range_str = f"{min_target}–{max_target}"
        target_pct_str = f"{min_pct:.0f}–{max_pct:.0f}%"
        gap = max(0, min_target - curr_cnt)

        print(f"{cap_name:<26} {curr_cnt:<9} {target_range_str:<15} {target_pct_str:<10} {gap:<6}")
        gap_report.append({
            "capability": cap_name,
            "current": curr_cnt,
            "target_min": min_target,
            "target_max": max_target,
            "target_pct": f"{min_pct}–{max_pct}%",
            "gap": gap
        })

    print("=" * 68 + "\n")
    return {
        "total_examples": total_examples,
        "target_total": target_total,
        "counts": counts,
        "gap_report": gap_report
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    train_file = base_dir / "data" / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.3.0.jsonl"
    if not train_file.exists():
        train_file = base_dir / "data" / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.2.1.jsonl"

    analyze_gaps(train_file, target_total=1000)


if __name__ == "__main__":
    main()
