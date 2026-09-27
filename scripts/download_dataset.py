#!/usr/bin/env python3
"""Download datasets from Hugging Face or other sources.

This script is intentionally conservative — it does NOT automatically download
datasets whose licensing or training permissions are unclear.

Usage:
    python scripts/download_dataset.py --list           # List registered datasets
    python scripts/download_dataset.py --name <name>    # Download a specific dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "metadata" / "dataset_registry.json"


def load_registry() -> list[dict]:
    """Load the dataset registry."""
    if not REGISTRY_PATH.exists():
        print(f"Registry not found: {REGISTRY_PATH}")
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def list_datasets() -> None:
    """Print all registered datasets."""
    registry = load_registry()
    print(f"\n{'='*60}")
    print("  REGISTERED DATASETS")
    print(f"{'='*60}")
    for entry in registry:
        training = entry.get("training_allowed")
        status = "✓" if training else ("?" if training is None else "✗")
        print(f"  [{status}] {entry['dataset_name']}")
        print(f"      Source:  {entry.get('source', 'N/A')}")
        print(f"      License: {entry.get('license', 'unknown')}")
        print(f"      Size:    {entry.get('size', 'unknown')}")
        print()
    print(f"{'='*60}")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download CyberCodeMini datasets")
    parser.add_argument("--list", action="store_true", help="List registered datasets")
    parser.add_argument("--name", type=str, help="Dataset name to download")
    parsed = parser.parse_args(args)

    if parsed.list:
        list_datasets()
        return 0

    if parsed.name:
        registry = load_registry()
        entry = next((e for e in registry if e["dataset_name"] == parsed.name), None)
        if entry is None:
            print(f"Dataset '{parsed.name}' not found in registry.")
            return 1

        if entry.get("training_allowed") is None:
            print(f"WARNING: Training permission for '{parsed.name}' is UNKNOWN.")
            print("This dataset will NOT be downloaded automatically.")
            print("Please verify the license and update the registry first.")
            return 1

        if not entry.get("training_allowed"):
            print(f"Training is NOT allowed for '{parsed.name}'.")
            return 1

        print(f"Dataset '{parsed.name}' is registered and training-allowed.")
        print("Download functionality will be implemented when external datasets are needed.")
        print("Currently using synthetic development data only.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
