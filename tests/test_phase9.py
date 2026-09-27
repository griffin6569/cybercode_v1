"""CyberCodeMini Phase 9 Unit Test Suite

Tests training pipeline components:
- Model loading, mode configuration (lora/qlora/full) & hardware detection
- Target module discovery & config validation
- PEFT LoRA configuration construction & parameter counts
- Quantization detection & graceful CPU fallback
- Reproducibility metadata generation
- Checkpoint discovery, state validation & resume verification
- Adapter export
- Full training pipeline execution
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from training.checkpointing import create_checkpoint, find_latest_checkpoint, validate_checkpoint
from training.export import export_adapter
from training.model import load_cybercode_model
from training.peft_config import build_lora_config, compute_trainable_parameters, validate_target_modules
from training.quantization import get_quantization_config
from training.train import run_training_pipeline
from training.utils import detect_hardware, generate_run_metadata, save_run_metadata, validate_training_config


class TestPhase9TrainingPipeline:

    def test_hardware_detection(self):
        hw = detect_hardware()
        assert hw.device in ("cuda", "mps", "cpu")
        assert isinstance(hw.python_version, str)

    def test_config_validation(self):
        valid_cfg = {
            "model": {"model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
            "training": {"dataset_path": "data/raw/dev_dataset.jsonl", "batch_size": 1, "learning_rate": 2e-4},
            "lora": {"enabled": True, "rank": 16, "alpha": 32},
        }
        # Should not raise exception
        validate_training_config(valid_cfg)

        invalid_cfg = {
            "model": {"model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
            "training": {"dataset_path": "data/raw/dev_dataset.jsonl", "batch_size": 0},  # Invalid batch size
        }
        with pytest.raises(ValueError, match="Invalid batch size"):
            validate_training_config(invalid_cfg)

    def test_peft_lora_config_construction(self):
        cfg = build_lora_config(enabled=True, rank=16, alpha=32, dropout=0.05)
        assert cfg is not None
        if hasattr(cfg, "r"):
            assert cfg.r == 16
            assert cfg.lora_alpha == 32
        else:
            assert cfg["r"] == 16

    def test_target_module_validation(self):
        class MockModel:
            def named_modules(self):
                return [("model.layers.0.self_attn.q_proj", None), ("model.layers.0.mlp.gate_proj", None)]

        model = MockModel()
        # Should pass
        validate_target_modules(model, ["q_proj", "gate_proj"])

        # Should fail with diagnostic error
        with pytest.raises(ValueError, match="were not found in the model architecture"):
            validate_target_modules(model, ["invalid_proj_name"])

    def test_quantization_fallback(self):
        # Request quantization when CUDA unavailable -> graceful fallback to cpu_fallback
        config, status = get_quantization_config(enabled=True, bits=4, compute_dtype="bfloat16")
        assert status.enabled is False or status.backend in ("bitsandbytes", "cpu_fallback")

    def test_run_metadata_generation(self, tmp_path):
        cfg = {
            "model": {"model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
            "training": {"dataset_path": "data/raw/dev_dataset.jsonl", "seed": 42},
            "lora": {"enabled": True, "rank": 16},
        }
        meta = generate_run_metadata(run_id="test_run_01", config=cfg)
        assert meta["run_id"] == "test_run_01"
        assert meta["training_hyperparameters"]["seed"] == 42

        saved_path = save_run_metadata(meta, output_path=tmp_path / "test_run_01.json")
        assert saved_path.exists()

    def test_checkpointing_and_validation(self, tmp_path):
        ckpt_dir = create_checkpoint(model=None, output_dir=tmp_path, step=5)
        assert ckpt_dir.exists()
        assert validate_checkpoint(ckpt_dir) is True

        latest = find_latest_checkpoint(tmp_path)
        assert latest == ckpt_dir

    def test_adapter_export(self, tmp_path):
        export_dir = tmp_path / "export_adapter"
        out_path = export_adapter(model=None, output_dir=export_dir, export_mode="adapter_only")
        assert out_path.exists()
        assert (out_path / "adapter_config.json").exists()

    def test_smoke_training_pipeline_execution(self):
        results = run_training_pipeline(
            config_path="configs/training.yaml",
            model_config_path="configs/model.yaml",
            smoke_test_override=True,
        )
        assert results["run_id"].startswith("cybercodemini")
        assert results["tokenized_train_count"] > 0
        assert Path(results["checkpoint_path"]).exists()
        assert Path(results["adapter_path"]).exists()
        assert Path(results["metadata_path"]).exists()
