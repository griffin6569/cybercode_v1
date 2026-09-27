"""CyberCodeMini Validate Imported Dataset CLI (Phase 10)

Validates an imported dataset file for schema errors, license approval, secrets, PII, and leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.validators.validator import validate_file


def main():
    parser = argparse.ArgumentParser(description="Validate an imported dataset JSONL file.")
    parser.add_argument("file_path", help="Path to JSONL dataset file")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return 1

    report = validate_file(str(file_path))
    print(report.summary())
    return 1 if report.has_critical_errors else 0


if __name__ == "__main__":
    sys.exit(main())
