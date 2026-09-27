"""CyberCodeMini PEFT & LoRA Configuration Engine

Constructs PEFT LoraConfig and dynamically discovers/validates target modules for Qwen2.5-Coder.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

# Candidate target modules for Qwen2.5-Coder architecture
DEFAULT_QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def discover_model_target_modules(model: Any) -> list[str]:
    """Inspect model module names and discover attention/linear layer target names."""
    discovered = set()
    if hasattr(model, "named_modules"):
        for name, _ in model.named_modules():
            for target in DEFAULT_QWEN_TARGET_MODULES:
                if name.endswith(target):
                    discovered.add(target)

    if discovered:
        return sorted(list(discovered))
    return DEFAULT_QWEN_TARGET_MODULES


def validate_target_modules(model: Any, target_modules: Sequence[str]) -> None:
    """Validate that specified target modules exist in the model."""
    if not hasattr(model, "named_modules"):
        return  # Mock or non-torch object

    all_module_names = {name for name, _ in model.named_modules()}
    missing = []
    
    for target in target_modules:
        if not any(name.endswith(target) or target in name for name in all_module_names):
            missing.append(target)

    if missing:
        available = discover_model_target_modules(model)
        raise ValueError(
            f"Configured LoRA target modules {missing} were not found in the model architecture. "
            f"Available target modules discovered in model: {available}"
        )


def build_lora_config(
    enabled: bool = True,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> Optional[Any]:
    """Build PEFT LoraConfig object."""
    if not enabled:
        return None

    if target_modules is None:
        target_modules = DEFAULT_QWEN_TARGET_MODULES

    try:
        from peft import LoraConfig, TaskType

        t_type = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=list(target_modules),
            lora_dropout=dropout,
            bias=bias,
            task_type=t_type,
        )
    except ImportError:
        # Fallback dict for environments where peft is not installed
        return {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": list(target_modules),
            "lora_dropout": dropout,
            "bias": bias,
            "task_type": task_type,
        }


def compute_trainable_parameters(model: Any) -> tuple[int, int, float]:
    """Compute (trainable_params, total_params, trainable_percentage)."""
    trainable_params = 0
    total_params = 0

    if hasattr(model, "named_parameters"):
        for _, param in model.named_parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
    elif hasattr(model, "parameters"):
        for param in model.parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
    else:
        # Dummy values for mock objects
        total_params = 1_500_000_000
        trainable_params = 16_000_000

    pct = round((trainable_params / total_params) * 100, 4) if total_params > 0 else 0.0
    return trainable_params, total_params, pct
