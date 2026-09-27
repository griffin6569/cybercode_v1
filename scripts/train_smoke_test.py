"""CyberCodeMini Phase 9 Smoke Training CLI Script

Executes a minimal, CPU-compatible 3-step local smoke training run:
1. Load validated dataset
2. Select deterministic small subset
3. Load tokenizer & model (or fallback mock)
4. Apply LoRA PEFT configuration
5. Collate batch & compute loss
6. Perform forward/backward/optimizer step
7. Evaluate
8. Save checkpoint & export adapter
9. Record reproducibility metadata
10. Verify artifacts & exit 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.checkpointing import find_latest_checkpoint, validate_checkpoint
from training.train import run_training_pipeline
from training.utils import detect_hardware


def main():
    print("Starting CyberCodeMini Phase 9 Local Smoke Test...")

    try:
        results = run_training_pipeline(
            config_path="configs/training.yaml",
            model_config_path="configs/model.yaml",
            smoke_test_override=True,
        )

        hw = detect_hardware()
        ckpt_path = Path(results["checkpoint_path"])
        adapter_path = Path(results["adapter_path"])
        metadata_path = Path(results["metadata_path"])

        # Verify artifacts exist
        ckpt_valid = validate_checkpoint(ckpt_path)
        adapter_valid = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
        metadata_valid = metadata_path.exists()

        if not (ckpt_valid and adapter_valid and metadata_valid):
            print("ERROR: Smoke test artifact verification failed!")
            print(f"  Checkpoint valid: {ckpt_valid}")
            print(f"  Adapter valid:    {adapter_valid}")
            print(f"  Metadata valid:   {metadata_valid}")
            return 1

        print("\n" + "=" * 60)
        print("  CyberCodeMini Phase 9 Smoke Test")
        print("=" * 60)
        print(f"  Model:                Qwen/Qwen2.5-Coder-1.5B-Instruct")
        print(f"  Method:               LoRA")
        print(f"  Quantization:         disabled")
        print(f"  Device:               {hw.device.upper()} ({hw.device_name})")
        print(f"  Train examples:       {results['tokenized_train_count']}")
        print(f"  Eval examples:        {results['tokenized_val_count']}")
        print(f"  Steps:                3")
        print("-" * 60)
        print(f"  Initial loss:         {results['initial_loss']}")
        print(f"  Final loss:           {results['final_loss']}")
        print(f"  Eval loss:            {results['eval_loss']}")
        print("-" * 60)
        print(f"  Checkpoint:           PASS ({ckpt_path.name})")
        print(f"  Adapter export:       PASS ({adapter_path.name})")
        print(f"  Metadata:             PASS ({metadata_path.name})")
        print("-" * 60)
        print(f"  Azure resources:      NONE")
        print(f"  Azure spend:          NONE")
        print("=" * 60)
        print("  RESULT: PASS")
        print("=" * 60 + "\n")
        return 0

    except Exception as err:
        print(f"\nFATAL: Smoke test failed with exception: {err}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
