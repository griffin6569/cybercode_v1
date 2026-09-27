#!/usr/bin/env python3
"""CyberCodeMini Smoke Test

Validates the local pipeline by:
1. Checking the dataset schema
2. Running dataset validation
3. Running deduplication
4. Running dataset splitting
5. Computing statistics
6. Optionally loading the configured model and running simple prompts

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-model   # Skip model loading (default)
    python scripts/smoke_test.py --with-model    # Include model loading
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _pass(msg: str) -> None:
    print(f"  ✓ PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ FAIL: {msg}")


def _info(msg: str) -> None:
    print(f"  ℹ {msg}")


def test_schema() -> bool:
    """Test that the schema module loads and validates correctly."""
    _section("SCHEMA VALIDATION")
    try:
        from datasets.schemas.schema import (
            Category,
            ExampleMetadata,
            Message,
            MessageRole,
            TrainingExample,
        )

        # Test valid example
        example = TrainingExample(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are an assistant."),
                Message(role=MessageRole.USER, content="Hello"),
                Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            ],
            metadata=ExampleMetadata(
                category=Category.CODE_GENERATION,
                synthetic=True,
                allowed_for_training=True,
            ),
        )
        assert example.messages[0].role == MessageRole.SYSTEM
        _pass("Valid example accepted")

        # Test invalid example (missing assistant)
        try:
            TrainingExample(
                messages=[
                    Message(role=MessageRole.USER, content="Hello"),
                ],
                metadata=ExampleMetadata(
                    category=Category.CODE_GENERATION,
                    synthetic=True,
                ),
            )
            _fail("Should have rejected example without assistant message")
            return False
        except Exception:
            _pass("Invalid example rejected (no assistant message)")

        # Test security example requires authorization
        try:
            TrainingExample(
                messages=[
                    Message(role=MessageRole.USER, content="Review this code"),
                    Message(role=MessageRole.ASSISTANT, content="Found SQL injection"),
                ],
                metadata=ExampleMetadata(
                    category=Category.SECURITY_REVIEW,
                    # authorization defaults to NOT_APPLICABLE
                    synthetic=True,
                ),
            )
            _fail("Should have rejected security example without authorization")
            return False
        except Exception:
            _pass("Security example without authorization rejected")

        return True
    except Exception as e:
        _fail(f"Schema test failed: {e}")
        return False


def test_validation() -> bool:
    """Test dataset validation on the dev dataset."""
    _section("DATASET VALIDATION")
    try:
        from datasets.validators.validator import validate_file

        dev_path = PROJECT_ROOT / "data" / "raw" / "dev_dataset.jsonl"
        if not dev_path.exists():
            _fail(f"Dev dataset not found: {dev_path}")
            return False

        report = validate_file(str(dev_path))
        print(report.summary())

        if report.total_examples == 0:
            _fail("No examples found in dev dataset")
            return False

        _info(f"Total: {report.total_examples}, Valid: {report.valid_examples}, "
              f"Invalid: {report.invalid_examples}")

        if report.has_critical_errors:
            _fail("Critical errors found in dev dataset")
            return False

        _pass(f"Dev dataset validated ({report.valid_examples}/{report.total_examples} valid)")
        return True
    except Exception as e:
        _fail(f"Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deduplication() -> bool:
    """Test deduplication on the dev dataset."""
    _section("DEDUPLICATION")
    try:
        from scripts.deduplicate import deduplicate

        input_path = PROJECT_ROOT / "data" / "raw" / "dev_dataset.jsonl"
        output_path = PROJECT_ROOT / "data" / "interim" / "deduped.jsonl"

        stats = deduplicate(input_path, output_path, method="exact")
        _info(f"Total: {stats['total_examples']}, Kept: {stats['kept']}, "
              f"Removed: {stats['duplicates_removed']}")

        if stats["kept"] == 0:
            _fail("Deduplication removed all examples")
            return False

        _pass(f"Deduplication complete ({stats['kept']} kept, "
              f"{stats['duplicates_removed']} removed)")

        # Also test normalized dedup
        output_path_norm = PROJECT_ROOT / "data" / "interim" / "deduped_normalized.jsonl"
        stats_norm = deduplicate(input_path, output_path_norm, method="normalized")
        _pass(f"Normalized dedup also works ({stats_norm['kept']} kept)")

        return True
    except Exception as e:
        _fail(f"Deduplication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split() -> bool:
    """Test dataset splitting."""
    _section("DATASET SPLITTING")
    try:
        from scripts.split_dataset import split_dataset

        input_path = PROJECT_ROOT / "data" / "interim" / "deduped.jsonl"
        output_dir = PROJECT_ROOT / "data"

        stats = split_dataset(input_path, output_dir, strategy="random", seed=42)
        _info(f"Total: {stats['total']}, Train: {stats['train']}, "
              f"Val: {stats['validation']}, Test: {stats['test']}")

        if stats["train"] == 0:
            _fail("Train split is empty")
            return False

        _pass(f"Random split complete (train={stats['train']}, "
              f"val={stats['validation']}, test={stats['test']})")

        return True
    except Exception as e:
        _fail(f"Split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics() -> bool:
    """Test dataset statistics computation."""
    _section("DATASET STATISTICS")
    try:
        from scripts.dataset_stats import compute_stats

        dev_path = PROJECT_ROOT / "data" / "raw" / "dev_dataset.jsonl"
        stats = compute_stats(dev_path)

        _info(f"Total examples: {stats['total_examples']}")
        _info(f"Categories: {stats['categories']}")
        _info(f"Languages: {stats['languages']}")
        _info(f"Synthetic: {stats['synthetic_percentage']}%")
        _info(f"Tool-use: {stats['tool_use_percentage']}%")

        if stats["total_examples"] == 0:
            _fail("No examples found for statistics")
            return False

        _pass(f"Statistics computed for {stats['total_examples']} examples")
        return True
    except Exception as e:
        _fail(f"Statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config() -> bool:
    """Test that all configuration files load correctly."""
    _section("CONFIGURATION")
    try:
        import yaml

        configs = ["model.yaml", "dataset.yaml", "training.yaml", "evaluation.yaml", "azure.yaml"]
        for config_name in configs:
            config_path = PROJECT_ROOT / "configs" / config_name
            if not config_path.exists():
                _fail(f"Config not found: {config_name}")
                return False

            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                _fail(f"Config is empty: {config_name}")
                return False
            _pass(f"Loaded {config_name}")

        # Verify azure.yaml safety settings
        with open(PROJECT_ROOT / "configs" / "azure.yaml", "r") as f:
            azure_config = yaml.safe_load(f)

        if not azure_config.get("require_manual_confirmation", False):
            _fail("azure.yaml: require_manual_confirmation must be true")
            return False
        _pass("Azure cost protection verified")

        return True
    except Exception as e:
        _fail(f"Config test failed: {e}")
        return False


def test_model(skip: bool = True) -> bool:
    """Optionally test model loading and inference."""
    _section("MODEL SMOKE TEST")
    if skip:
        _info("Skipped (use --with-model to enable)")
        return True

    try:
        import yaml

        with open(PROJECT_ROOT / "configs" / "model.yaml", "r") as f:
            model_config = yaml.safe_load(f)

        model_name = model_config["model_name"]
        _info(f"Attempting to load: {model_name}")

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _info(f"Device: {device}")
            if device == "cpu":
                _info("WARNING: Running on CPU will be slow")
        except ImportError:
            _info("PyTorch not installed, skipping model test")
            return True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_config["tokenizer_name"],
                trust_remote_code=model_config.get("trust_remote_code", False),
            )

            _info("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=model_config.get("trust_remote_code", False),
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                device_map=device,
            )

            # Simple test prompt
            prompts = [
                "Write a Python function to reverse a string.",
                "What is SQL injection?",
            ]

            for prompt in prompts:
                _info(f"Testing prompt: {prompt[:50]}...")
                start = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=100, do_sample=False
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elapsed = time.time() - start
                _info(f"Response ({elapsed:.1f}s): {response[:100]}...")

            _pass("Model inference working")
            return True

        except ImportError:
            _info("transformers not installed, skipping model test")
            return True
        except Exception as e:
            _fail(f"Model loading/inference failed: {e}")
            return False

    except Exception as e:
        _fail(f"Model test failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="CyberCodeMini Smoke Test")
    parser.add_argument("--with-model", action="store_true", help="Include model loading test")
    parser.add_argument("--skip-model", action="store_true", default=True, help="Skip model test (default)")
    args = parser.parse_args()

    skip_model = not args.with_model

    print("\n" + "=" * 60)
    print("  CYBERCODEMINI SMOKE TEST")
    print("=" * 60)

    results = {
        "Schema": test_schema(),
        "Validation": test_validation(),
        "Deduplication": test_deduplication(),
        "Splitting": test_split(),
        "Statistics": test_statistics(),
        "Configuration": test_config(),
        "Model": test_model(skip=skip_model),
    }

    _section("RESULTS SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
