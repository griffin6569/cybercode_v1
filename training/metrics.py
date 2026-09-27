"""CyberCodeMini Training Metrics Tracker

Records machine-readable training metrics (loss, eval_loss, learning_rate, steps, tokens, elapsed time).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MetricStep:
    step: int
    epoch: float
    loss: float
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    tokens_processed: int = 0
    timestamp: float = field(default_factory=time.time)


class TrainingMetricsTracker:
    """Tracks and serializes metrics to config.json, metrics.json, and trainer_state.json."""

    def __init__(self, run_dir: Optional[Path | str] = None) -> None:
        self.run_dir = Path(run_dir) if run_dir else Path("outputs/runs/default_run")
        self.steps: list[MetricStep] = []
        self.start_time = time.time()
        self.initial_loss: Optional[float] = None
        self.final_loss: Optional[float] = None
        self.eval_loss: Optional[float] = None

    def log_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        eval_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_processed: int = 0,
    ) -> None:
        m = MetricStep(
            step=step,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_loss,
            learning_rate=learning_rate,
            tokens_processed=tokens_processed,
        )
        self.steps.append(m)

        if self.initial_loss is None:
            self.initial_loss = loss
        self.final_loss = loss
        if eval_loss is not None:
            self.eval_loss = eval_loss

    def to_dict(self) -> dict[str, Any]:
        elapsed = round(time.time() - self.start_time, 2)
        total_tokens = sum(s.tokens_processed for s in self.steps)
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "eval_loss": self.eval_loss,
            "total_steps": len(self.steps),
            "total_tokens_processed": total_tokens,
            "training_time_seconds": elapsed,
            "history": [asdict(s) for s in self.steps],
        }

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return metrics_file
