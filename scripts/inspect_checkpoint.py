"""CyberCodeMini Checkpoint Inspector CLI

Inspects a checkpoint directory and validates state files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.checkpointing import find_latest_checkpoint, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate a training checkpoint.")
    parser.add_argument("checkpoint", nargs="?", default="latest", help="Path to checkpoint dir or 'latest'")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint != "latest" else None

    if ckpt_path is None or not ckpt_path.exists():
        ckpt_dir = Path("outputs/checkpoints")
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    if ckpt_path is None or not ckpt_path.exists():
        print("Error: No valid checkpoint directory found.")
        return 1

    valid = validate_checkpoint(ckpt_path)
    contents = [item.name for item in ckpt_path.iterdir()]

    print("\n" + "=" * 60)
    print("  CYBERCODEMINI CHECKPOINT INSPECTION")
    print("=" * 60)
    print(f"  Path:     {ckpt_path.resolve()}")
    print(f"  Status:   {'VALID' if valid else 'INVALID / INCOMPLETE'}")
    print(f"  Files:    {', '.join(contents)}")
    print("=" * 60 + "\n")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
