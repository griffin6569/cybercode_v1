"""Unit tests for CyberCodeMini training configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTrainingConfig:
    def test_smoke_test_mode_available(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        assert "smoke_test" in config
        assert "enabled" in config["smoke_test"]
        assert "max_steps" in config["smoke_test"]
        assert "label" in config["smoke_test"]
        assert config["smoke_test"]["label"] == "SMOKE_TEST"

    def test_lora_config_reasonable(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        # LoRA rank should be reasonable for a 1.5B model
        assert 4 <= config["lora_r"] <= 64
        assert config["lora_alpha"] >= config["lora_r"]
        assert 0 <= config["lora_dropout"] < 0.5

    def test_learning_rate_conservative(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        # For LoRA fine-tuning, LR should be in reasonable range
        assert 1e-5 <= config["learning_rate"] <= 1e-3

    def test_gradient_checkpointing_enabled(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        assert config["gradient_checkpointing"] is True, \
            "Gradient checkpointing should be on to save GPU memory"

    def test_seed_is_set(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        assert isinstance(config["seed"], int)
