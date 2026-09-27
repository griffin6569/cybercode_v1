#!/usr/bin/env python3
"""Azure preflight check for CyberCodeMini.

Displays training configuration, estimated costs, and requires
explicit human confirmation before any billable operation.

Usage:
    python azure/scripts/preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    import yaml

    print("\n" + "=" * 60)
    print("  CYBERCODEMINI AZURE PREFLIGHT CHECK")
    print("=" * 60)

    # Load configs
    with open(PROJECT_ROOT / "configs" / "azure.yaml") as f:
        azure_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
        training_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "model.yaml") as f:
        model_config = yaml.safe_load(f)

    # Check manual confirmation
    if not azure_config.get("require_manual_confirmation", True):
        print("\n  ⚠️  SAFETY CHECK FAILED")
        print("  require_manual_confirmation is not set to true in azure.yaml")
        print("  Refusing to proceed.")
        return 1

    # Display configuration
    print(f"\n  Model:              {model_config['model_name']}")
    print(f"  Compute:            {azure_config.get('compute_name', 'NOT SET')}")
    print(f"  VM Size:            {azure_config.get('preferred_vm_size', 'NOT SET')}")
    print(f"  Max Runtime:        {azure_config.get('max_runtime_hours', '?')} hours")
    print(f"  Auto Shutdown:      {azure_config.get('auto_shutdown', False)}")
    print(f"  Epochs:             {training_config.get('epochs', '?')}")
    print(f"  Batch Size:         {training_config.get('batch_size', '?')}")
    print(f"  LoRA Rank:          {training_config.get('lora_r', '?')}")
    print(f"  Max Cost/Run:       ${azure_config.get('max_cost_per_run_usd', '?')}")

    # Check dataset
    train_path = PROJECT_ROOT / "data" / "train" / "train.jsonl"
    if train_path.exists():
        with open(train_path) as f:
            n_examples = sum(1 for _ in f)
        print(f"  Training Examples:  {n_examples}")
    else:
        print("  Training Examples:  NOT FOUND")
        print("\n  ⚠️  No training data found. Run the dataset pipeline first.")
        return 1

    # Subscription check
    sub_id = azure_config.get("subscription_id", "")
    if not sub_id:
        print("\n  ⚠️  Azure subscription_id not set in configs/azure.yaml or .env")
        print("  Set this before submitting jobs.")

    print("\n" + "-" * 60)
    print("  This operation will incur Azure costs.")
    print("-" * 60)

    response = input("\n  Type 'CONFIRM' to proceed: ").strip()
    if response != "CONFIRM":
        print("\n  Cancelled. No Azure resources created.")
        return 1

    print("\n  ✓ Confirmed. You may now submit the training job.")
    print("  Run: az ml job create --file azure/train_job.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
