"""CyberCodeMini Custom Training Callbacks

Logging and diagnostic callbacks for Hugging Face Trainer.
"""

from __future__ import annotations

from typing import Any, Optional


class LoggingCallback:
    """Custom callback to track loss metrics and step stats."""

    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs:
            step_log = dict(logs)
            step_log["step"] = getattr(state, "global_step", len(self.logs))
            self.logs.append(step_log)
