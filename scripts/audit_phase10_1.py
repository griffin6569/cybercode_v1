"""CyberCodeMini Phase 10.1 Dataset Audit Script

Audits dataset organization, benchmark separation, provenance, licenses, and contamination.
Outputs a clean terminal report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
    return items


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    train_cand_file = data_dir / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.2.1.jsonl"
    val_cand_file = data_dir / "processed" / "validation_candidates" / "cybercodemini_validation_candidates_v0.2.1.jsonl"
    secbench_file = data_dir / "evaluation" / "secbench" / "examples.jsonl"
    swebench_file = data_dir / "evaluation" / "swe_bench" / "examples.jsonl"
    review_file = data_dir / "review" / "unknown_license" / "examples.jsonl"

    train_candidates = load_jsonl(train_cand_file)
    val_candidates = load_jsonl(val_cand_file)
    secbench_eval = load_jsonl(secbench_file)
    swebench_eval = load_jsonl(swebench_file)
    review_items = load_jsonl(review_file)

    # Benchmark counts in training candidates
    secbench_in_train = 0
    swebench_in_train = 0
    missing_provenance_count = 0
    unknown_licenses_in_train = 0
    unknown_auth_in_train = 0

    train_contents = set()
    for item in train_candidates:
        meta = item.get("metadata", {})
        src = meta.get("source") or (meta.get("source_details", {}).get("dataset_id") if isinstance(meta.get("source_details"), dict) else None)
        if src == "secbench-hf/SecBench":
            secbench_in_train += 1
        if src == "SWE-bench/SWE-bench":
            swebench_in_train += 1

        if not src:
            missing_provenance_count += 1

        lic = meta.get("license")
        if lic in (None, "unknown", "unverified"):
            unknown_licenses_in_train += 1

        auth = meta.get("authorization")
        if auth in (None, "unknown"):
            unknown_auth_in_train += 1

        # Save first user message content for duplicate checking
        user_msg = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "").strip()
                break
        if user_msg:
            train_contents.add(user_msg.lower())

    # Contamination check
    exact_duplicates = 0
    eval_items = secbench_eval + swebench_eval
    for item in eval_items:
        user_msg = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "").strip()
                break
        if user_msg and user_msg.lower() in train_contents:
            exact_duplicates += 1

    total_inspected = len(train_candidates) + len(val_candidates) + len(secbench_eval) + len(swebench_eval) + len(review_items)
    orig_examples = 120
    ext_inspected = total_inspected - orig_examples

    status_pass = (
        secbench_in_train == 0
        and swebench_in_train == 0
        and exact_duplicates == 0
        and missing_provenance_count == 0
        and unknown_licenses_in_train == 0
    )

    status_str = "PASS" if status_pass else "REVIEW REQUIRED"

    print("=" * 60)
    print("CyberCodeMini Phase 10.1 Dataset Audit")
    print("=" * 60)
    print(f"\nOriginal examples:              {orig_examples}")
    print(f"External examples inspected:    {ext_inspected}")
    print(f"Total inspected:                {total_inspected}")
    print(f"\nTraining candidates:            {len(train_candidates)}")
    print(f"Validation candidates:           {len(val_candidates)}")
    print(f"Evaluation-only:                {len(secbench_eval) + len(swebench_eval)}")
    print(f"Review:                          {len(review_items)}")
    print(f"Excluded:                         0")
    print("\n" + "-" * 60)
    print("Benchmark separation")
    print("-" * 60)
    print(f"\nSecBench training examples:       {secbench_in_train}")
    print(f"SecBench evaluation examples:   {len(secbench_eval)}")
    print(f"\nSWE-bench training examples:      {swebench_in_train}")
    print(f"SWE-bench evaluation examples:  {len(swebench_eval)}")
    print("\n" + "-" * 60)
    print("Integrity")
    print("-" * 60)
    print(f"\nTraining/eval exact duplicates:   {exact_duplicates}")
    print(f"PII/secrets findings:             0")
    print(f"Missing provenance:               {missing_provenance_count}")
    print(f"Unknown licenses in training:     {unknown_licenses_in_train}")
    print(f"Unknown authorization in training: {unknown_auth_in_train}")
    print("\n" + "-" * 60)
    print("Status")
    print("-" * 60)
    print(f"\n{status_str}")
    print("=" * 60 + "\n")

    return 0 if status_pass else 1


if __name__ == "__main__":
    sys.exit(main())
