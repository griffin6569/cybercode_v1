"""CyberCodeMini Dataset Ingestion CLI (Phase 7)

Ingests external datasets using specified adapter format, enforces registry permission,
preserves source IDs, and outputs canonical CyberCodeMini JSONL format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.adapters.ctf_lab_adapter import CTFLabAdapter
from cybercode_datasets.adapters.generic_instruction_adapter import GenericInstructionAdapter
from cybercode_datasets.adapters.hf_conversational_adapter import HuggingFaceConversationalAdapter
from cybercode_datasets.adapters.security_review_adapter import SecurityReviewAdapter
from cybercode_datasets.adapters.trajectory_adapter import AgentTrajectoryAdapter
from cybercode_datasets.provenance.provenance import DatasetRegistry


def get_adapter(adapter_name: str, source_name: str, registry: DatasetRegistry):
    name = adapter_name.lower()
    if name in ("hf", "huggingface"):
        return HuggingFaceConversationalAdapter(source_name=source_name, registry=registry)
    elif name in ("sec", "security", "security_review"):
        return SecurityReviewAdapter(source_name=source_name, registry=registry)
    elif name in ("traj", "trajectory", "agent"):
        return AgentTrajectoryAdapter(source_name=source_name, registry=registry)
    elif name in ("ctf", "lab", "ctf_lab"):
        return CTFLabAdapter(source_name=source_name, registry=registry)
    elif name in ("generic", "instruction"):
        return GenericInstructionAdapter(source_name=source_name, registry=registry)
    else:
        raise ValueError(f"Unknown adapter type: '{adapter_name}'. Choose from: hf, sec, traj, ctf, generic")


def main():
    parser = argparse.ArgumentParser(description="Ingest an external dataset using a CyberCodeMini adapter.")
    parser.add_argument("--input", "-i", required=True, help="Path to input raw JSON or JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSONL file")
    parser.add_argument("--adapter", "-a", required=True, help="Adapter type: hf, sec, traj, ctf, generic")
    parser.add_argument("--source-name", "-s", required=True, help="Registered dataset source name")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return 1

    registry = DatasetRegistry()
    adapter = get_adapter(args.adapter, args.source_name, registry)

    # Read raw entries
    raw_entries = []
    if input_path.suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_entries.append(json.loads(line))
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_entries = data
            elif isinstance(data, dict):
                raw_entries = [data]

    print(f"Ingesting {len(raw_entries)} raw entries from '{input_path}' using adapter '{adapter.__class__.__name__}'...")
    
    converted_examples = adapter.convert_batch(raw_entries)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in converted_examples:
            f.write(json.dumps(ex.model_dump()) + "\n")

    print(f"Successfully converted and saved {len(converted_examples)} examples to '{output_path}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
