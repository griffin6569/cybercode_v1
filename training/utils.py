"""CyberCodeMini Training Utilities & Hardware Detection

Provides hardware environment detection, training configuration validation,
and reproducibility metadata generation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class HardwareInfo:
    """Hardware environment details."""

    device: str  # "cuda" | "mps" | "cpu"
    device_count: int = 1
    device_name: str = "CPU"
    total_memory_gb: float = 0.0
    python_version: str = field(default_factory=platform.python_version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    """Detect available compute hardware (CUDA, MPS, or CPU)."""
    device = "cpu"
    device_count = 1
    device_name = platform.processor() or "Generic CPU"
    memory_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon MPS"
    except ImportError:
        pass

    return HardwareInfo(
        device=device,
        device_count=device_count,
        device_name=device_name,
        total_memory_gb=memory_gb,
    )


def get_installed_versions() -> dict[str, str]:
    """Retrieve installed versions of core ML libraries."""
    versions = {
        "python": platform.python_version(),
    }
    
    for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not_installed"

    return versions


def validate_training_config(config: dict[str, Any]) -> None:
    """Validate training configuration against impossible or dangerous settings."""
    model_cfg = config.get("model", config)
    training_cfg = config.get("training", config)
    lora_cfg = config.get("lora", config.get("lora_config", {}))

    # 1. Model & Dataset checks
    model_name = model_cfg.get("model_name") or model_cfg.get("base_model") or config.get("base_model")
    if not model_name:
        raise ValueError("Training configuration missing required field: 'model_name' or 'base_model'")

    dataset_path = training_cfg.get("dataset_path") or config.get("dataset_path")
    if not dataset_path:
        raise ValueError("Training configuration missing required field: 'dataset_path'")

    # 2. Hyperparameter positivity checks
    batch_size = training_cfg.get("batch_size", training_cfg.get("per_device_train_batch_size", 1))
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"Invalid batch size ({batch_size}). Must be > 0.")

    lr = training_cfg.get("learning_rate", 2e-4)
    if lr <= 0:
        raise ValueError(f"Invalid learning rate ({lr}). Must be > 0.")

    max_seq_len = training_cfg.get("max_seq_length") or model_cfg.get("max_sequence_length", 2048)
    if max_seq_len <= 0:
        raise ValueError(f"Invalid max_seq_length ({max_seq_len}). Must be > 0.")

    # 3. LoRA checks
    if lora_cfg.get("enabled", True):
        rank = lora_cfg.get("rank") or lora_cfg.get("lora_r", 16)
        if rank <= 0:
            raise ValueError(f"Invalid LoRA rank ({rank}). Must be > 0.")

        alpha = lora_cfg.get("alpha") or lora_cfg.get("lora_alpha", 32)
        if alpha <= 0:
            raise ValueError(f"Invalid LoRA alpha ({alpha}). Must be > 0.")


def generate_run_metadata(
    run_id: str,
    config: dict[str, Any],
    dataset_hash: str = "unknown",
    dataset_manifest_version: str = "1.0.0",
) -> dict[str, Any]:
    """Generate complete reproducibility metadata dictionary."""
    model_cfg = config.get("model", config)
    training_cfg = config.get("training", config)
    lora_cfg = config.get("lora", config.get("lora_config", {}))
    quant_cfg = config.get("quantization", {})

    hw = detect_hardware()
    versions = get_installed_versions()

    # Compute deterministic config hash
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]

    metadata = {
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "dataset_manifest_version": dataset_manifest_version,
        "model": {
            "model_name": model_cfg.get("model_name") or config.get("base_model", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "model_revision": model_cfg.get("model_revision", "main"),
            "tokenizer_name": model_cfg.get("tokenizer_name", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "max_sequence_length": model_cfg.get("max_sequence_length", 2048),
        },
        "method": "qlora" if quant_cfg.get("enabled", False) else ("lora" if lora_cfg.get("enabled", True) else "full"),
        "training_hyperparameters": {
            "seed": training_cfg.get("seed", 42),
            "epochs": training_cfg.get("epochs") or training_cfg.get("num_train_epochs", 1),
            "batch_size": training_cfg.get("batch_size") or training_cfg.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
            "learning_rate": training_cfg.get("learning_rate", 2e-4),
            "warmup_ratio": training_cfg.get("warmup_ratio", 0.03),
            "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        },
        "lora_configuration": lora_cfg,
        "quantization_configuration": quant_cfg,
        "hardware": hw.to_dict(),
        "library_versions": versions,
    }

    return metadata


def save_run_metadata(metadata: dict[str, Any], output_path: Optional[Path | str] = None) -> Path:
    """Save reproducibility metadata to data/metadata/training_runs/<run_id>.json."""
    run_id = metadata.get("run_id", "default_run")
    if output_path is None:
        output_path = Path("data/metadata/training_runs") / f"{run_id}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path
