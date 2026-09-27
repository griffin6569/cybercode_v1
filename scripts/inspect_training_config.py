"""CyberCodeMini Training Config Inspector CLI

Validates and displays training and model configurations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.utils import detect_hardware, validate_training_config


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate training configuration.")
    parser.add_argument("--config", "-c", default="configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--model-config", "-m", default="configs/model.yaml", help="Path to model.yaml")
    args = parser.parse_args()

    train_cfg_path = Path(args.config)
    model_cfg_path = Path(args.model_config)

    if not train_cfg_path.exists():
        print(f"Error: Config '{train_cfg_path}' not found.")
        return 1

    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    full_cfg = {"model": model_cfg, "training": train_cfg}

    print("\n" + "=" * 60)
    print("  CYBERCODEMINI TRAINING CONFIGURATION INSPECTION")
    print("=" * 60)
    
    try:
        validate_training_config(full_cfg)
        print("  Status:               VALIDATED")
    except ValueError as err:
        print(f"  Status:               INVALID ({err})")
        return 1

    hw = detect_hardware()
    print(f"  Model:                {model_cfg.get('model_name')}")
    print(f"  Tokenizer:            {model_cfg.get('tokenizer_name')}")
    print(f"  Max Sequence Length:  {train_cfg.get('max_seq_length', 2048)}")
    print(f"  Learning Rate:        {train_cfg.get('learning_rate')}")
    print(f"  Batch Size:           {train_cfg.get('batch_size')}")
    print(f"  Grad Accum Steps:     {train_cfg.get('gradient_accumulation_steps')}")
    print(f"  LoRA Enabled:         {train_cfg.get('lora_config', {}).get('enabled', True)}")
    print(f"  LoRA Rank:            {train_cfg.get('lora_config', {}).get('rank')}")
    print(f"  Loss Masking:         {train_cfg.get('loss_masking', {}).get('strategy')}")
    print(f"  Device Detected:      {hw.device.upper()} ({hw.device_name})")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
