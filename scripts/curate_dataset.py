"""CyberCodeMini Dataset Curation & Manifest Generator CLI (Phase 7)

Applies quality filters, contamination checks, deduplication, routes flagged items
to data/metadata/review_queue.jsonl, and generates data/metadata/manifest.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.filters.quality_filter import QualityFilter
from datasets.provenance.provenance import DatasetRegistry
from datasets.schemas.schema import TrainingExample


def generate_manifest(
    input_file: Path,
    approved_examples: list[TrainingExample],
    rejected_count: int,
    review_count: int,
    manifest_path: Path,
) -> dict:
    registry = DatasetRegistry()

    total = len(approved_examples)
    categories: dict[str, int] = {}
    sources: dict[str, int] = {}
    licenses: dict[str, int] = {}
    synthetic_count = 0
    external_count = 0

    hashes = []

    for ex in approved_examples:
        m = ex.metadata
        cat = m.category.value if hasattr(m.category, "value") else str(m.category)
        categories[cat] = categories.get(cat, 0) + 1

        src = m.source or "unknown"
        sources[src] = sources.get(src, 0) + 1

        # Lookup license from registry
        reg_entry = registry.get_entry(src)
        lic = reg_entry.license if reg_entry else "unknown"
        licenses[lic] = licenses.get(lic, 0) + 1

        if m.synthetic:
            synthetic_count += 1
        else:
            external_count += 1

        # SHA256 of message content
        ex_bytes = json.dumps(ex.model_dump()).encode("utf-8")
        hashes.append(DatasetRegistry.compute_sha256(ex_bytes))

    manifest_data = {
        "dataset_version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_file": str(input_file),
        "total_examples": total,
        "categories": categories,
        "sources": sources,
        "licenses": licenses,
        "synthetic_count": synthetic_count,
        "external_count": external_count,
        "review_required": review_count,
        "approved_count": total,
        "rejected_count": rejected_count,
        "sha256_hashes": hashes[:50],  # Sample first 50 hashes
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Curate dataset, run quality filters, and update manifest.")
    parser.add_argument("--input", "-i", default="data/raw/dev_dataset.jsonl", help="Input dataset path")
    parser.add_argument("--output", "-o", default="data/processed/curated_dev.jsonl", help="Curated output path")
    parser.add_argument("--min-score", type=float, default=70.0, help="Minimum quality score (0-100)")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path("data/metadata/manifest.json")

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return 1

    print(f"Curating dataset '{input_path}'...")
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(TrainingExample.model_validate(data))

    quality_filter = QualityFilter(min_quality_score=args.min_score)
    approved, rejected = quality_filter.filter_batch(examples)

    review_queue_path = Path("data/metadata/review_queue.jsonl")
    review_count = 0
    if review_queue_path.exists():
        with open(review_queue_path, "r", encoding="utf-8") as f:
            review_count = sum(1 for _ in f if _.strip())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in approved:
            f.write(json.dumps(ex.model_dump()) + "\n")

    manifest = generate_manifest(
        input_file=input_path,
        approved_examples=approved,
        rejected_count=len(rejected),
        review_count=review_count,
        manifest_path=manifest_path,
    )

    print(f"Curation complete!")
    print(f"  Approved:      {manifest['approved_count']}")
    print(f"  Rejected:      {manifest['rejected_count']}")
    print(f"  Review Queue:  {manifest['review_required']}")
    print(f"  Manifest written to: '{manifest_path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
