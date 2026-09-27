"""CyberCodeMini Learning Rate Scheduler Helper

Constructs learning rate schedulers (cosine, linear, constant with warmup).
"""

from __future__ import annotations

from typing import Any, Optional


def build_scheduler(
    optimizer: Any,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "cosine",
) -> Any:
    """Build learning rate scheduler."""
    if optimizer is None:
        return None

    try:
        from transformers import get_scheduler
        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    except Exception:
        return None
