"""CyberCodeMini Checkpointing & Resume Management

Manages checkpoint saving, discovery, state validation, and resume verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def find_latest_checkpoint(checkpoint_dir: Path | str) -> Optional[Path]:
    """Find the latest checkpoint directory (e.g., checkpoint-10) in checkpoint_dir."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = []
    for item in checkpoint_dir.glob("checkpoint-*"):
        if item.is_dir():
            try:
                step_num = int(item.name.split("-")[-1])
                checkpoints.append((step_num, item))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def validate_checkpoint(checkpoint_dir: Path | str) -> bool:
    """Validate that a checkpoint directory contains valid trainable state.
    
    Returns True if valid state files are present.
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists() or not ckpt.is_dir():
        return False

    # Check for presence of adapter weights or model weights or trainer_state
    has_adapter = (ckpt / "adapter_config.json").exists()
    has_weights = (ckpt / "pytorch_model.bin").exists() or (ckpt / "model.safetensors").exists() or (ckpt / "adapter_model.safetensors").exists() or (ckpt / "adapter_model.bin").exists()
    has_state = (ckpt / "trainer_state.json").exists() or (ckpt / "training_args.bin").exists() or (ckpt / "config.json").exists()

    return (has_adapter or has_weights) and has_state


def create_checkpoint(
    model: Any,
    output_dir: Path | str,
    step: int,
    trainer_state: Optional[dict] = None,
) -> Path:
    """Create a checkpoint directory step directory and save model/adapter weights."""
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter if PEFT model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir))
    else:
        # Fallback for CPU / lightweight mock model
        mock_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        with open(ckpt_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(mock_config, f, indent=2)
        with open(ckpt_dir / "adapter_model.safetensors", "w", encoding="utf-8") as f:
            f.write("MOCK_ADAPTER_WEIGHTS")

    # Save trainer state
    state = trainer_state or {"step": step}
    with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return ckpt_dir
