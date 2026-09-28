"""CyberCodeMini Ingest Hugging Face Dataset CLI (Phase 10)

Ingests, inspects, samples, and converts Hugging Face dataset candidates using pinned revisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.ingestion.downloader import SafeDatasetDownloader
from cybercode_datasets.ingestion.huggingface import HuggingFaceIngestionEngine


def main():
    parser = argparse.ArgumentParser(description="Ingest a Hugging Face dataset candidate.")
    parser.add_argument("dataset_id", help="Hugging Face dataset identifier (e.g. secbench-hf/SecBench)")
    parser.add_argument("--revision", default="main", help="Pinned revision commit/tag")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--license", default="MIT", help="Declared license")
    parser.add_argument("--input-file", help="Optional local JSON/JSONL sample file")
    parser.add_argument("--max-rows", type=int, default=500, help="Max rows to ingest")
    parser.add_argument("--output", "-o", help="Output JSONL path")
    args = parser.parse_args()

    engine = HuggingFaceIngestionEngine()
    downloader = SafeDatasetDownloader()

    # Load raw sample rows
    if args.input_file:
        raw_rows = downloader.fetch_local_or_cached_jsonl(args.input_file, max_rows=args.max_rows)
    else:
        # Mock sample rows for candidate inspection
        raw_rows = [
            {"id": f"{args.dataset_id.replace('/', '_')}_{i}", "instruction": f"Security query #{i}", "response": f"Secure analysis #{i}"}
            for i in range(1, min(50, args.max_rows + 1))
        ]

    converted_examples, inspection = engine.ingest_raw_rows(
        dataset_id=args.dataset_id,
        raw_rows=raw_rows,
        revision=args.revision,
        split=args.split,
        license_name=args.license,
        license_source="cli_inspection",
        max_rows=args.max_rows,
    )

    out_path = Path(args.output) if args.output else Path("data/interim") / f"{args.dataset_id.replace('/', '_')}_ingested.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in converted_examples:
            f.write(json.dumps(ex.model_dump()) + "\n")

    print("\n" + "=" * 60)
    print(f"  INGESTION COMPLETE: {args.dataset_id}")
    print("=" * 60)
    print(f"  Rows Sampled:         {len(sampled_rows := raw_rows[:args.max_rows])}")
    print(f"  Examples Converted:   {len(converted_examples)}")
    print(f"  Approval Status:      {inspection['approval_status']}")
    print(f"  Output Path:          {out_path}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
