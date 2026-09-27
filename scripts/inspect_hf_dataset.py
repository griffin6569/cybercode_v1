"""CyberCodeMini Inspect Hugging Face Dataset CLI (Phase 10)

Inspects candidate Hugging Face dataset cards, revisions, and license compliance.
Outputs inspection report to data/metadata/ingestion_reports/<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.ingestion.huggingface import HuggingFaceIngestionEngine


def main():
    parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset candidate.")
    parser.add_argument("dataset_id", help="Hugging Face dataset identifier (e.g. SWE-bench/SWE-bench or secbench-hf/SecBench)")
    parser.add_argument("--revision", default="main", help="Pinned commit/tag revision")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--license", default="unknown", help="Declared dataset license")
    args = parser.parse_args()

    engine = HuggingFaceIngestionEngine()
    report = engine.inspect_dataset(
        dataset_id=args.dataset_id,
        revision=args.revision,
        split=args.split,
        license_name=args.license,
        license_source="dataset_card_inspection",
    )

    print("\n" + "=" * 60)
    print(f"  HUGGING FACE DATASET INSPECTION REPORT: {args.dataset_id}")
    print("=" * 60)
    print(f"  Revision:             {report['revision']}")
    print(f"  Split:                {report['split']}")
    print(f"  Declared License:     {report['license']}")
    print(f"  License Verified:     {report['license_verified']}")
    print(f"  Commercial Restr:     {report['commercial_restriction']}")
    print(f"  Approval Status:      {report['approval_status']}")
    print(f"  Training Candidate:   {report['training_candidate']}")
    print(f"  Evaluation Candidate: {report['evaluation_candidate']}")
    print(f"  Notes:                {report['notes']}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
