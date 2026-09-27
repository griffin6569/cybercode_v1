"""CyberCodeMini Adapter Export CLI Tool

Exports model adapter or merged model weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.export import export_adapter
from training.model import load_cybercode_model


def main():
    parser = argparse.ArgumentParser(description="Export CyberCodeMini adapter weights.")
    parser.add_argument("--output-dir", "-o", default="outputs/adapters/cybercodemini-qwen2.5-coder-1.5b", help="Export path")
    parser.add_argument("--mode", choices=["adapter_only", "merged", "quantized"], default="adapter_only", help="Export mode")
    args = parser.parse_args()

    print(f"Exporting model in '{args.mode}' mode to '{args.output_dir}'...")

    model, _ = load_cybercode_model(mode="lora")
    out_path = export_adapter(model, output_dir=args.output_dir, export_mode=args.mode)

    print(f"Export complete: '{out_path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
