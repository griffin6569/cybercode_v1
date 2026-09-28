#!/usr/bin/env python3
"""Validate a CyberCodeMini JSONL dataset file.

Usage:
    python scripts/validate_dataset.py data/raw/dev_dataset.jsonl
    python scripts/validate_dataset.py data/raw/dev_dataset.jsonl --json
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cybercode_datasets.validators.validator import validate_file_cli

if __name__ == "__main__":
    sys.exit(validate_file_cli())
