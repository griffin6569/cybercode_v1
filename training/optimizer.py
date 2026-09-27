"""CyberCodeMini Optimizer Helper

Constructs PyTorch or Hugging Face optimizers for fine-tuning.
"""

from __future__ import annotations

from typing import Any, Optional


def build_optimizer(
    model: Any,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw_torch",
) -> Any:
    """Build optimizer instance for model trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad] if hasattr(model, "parameters") else []
    
    try:
        import torch
        if optimizer_type in ("adamw_8bit", "paged_adamw_8bit"):
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
            except Exception:
                pass
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    except ImportError:
        return None
