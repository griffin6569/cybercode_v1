"""Unit tests for CyberCodeMini dataset schema, validation, deduplication, and splitting."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.schemas.schema import (
    Authorization,
    Category,
    Difficulty,
    Environment,
    ExampleMetadata,
    Message,
    MessageRole,
    ToolCall,
    TrainingExample,
)


# ── Schema Tests ──────────────────────────────────────────────


class TestMessage:
    def test_valid_message(self):
        msg = Message(role=MessageRole.USER, content="Hello world")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"

    def test_empty_content_rejected(self):
        with pytest.raises(Exception):
            Message(role=MessageRole.USER, content="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(Exception):
            Message(role=MessageRole.USER, content="   \n  ")

    def test_tool_calls_only_on_assistant(self):
        # Should work on assistant
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Let me call a tool",
            tool_calls=[ToolCall(name="read_file", arguments={"path": "test.py"})],
        )
        assert msg.tool_calls is not None

        # Should fail on user
        with pytest.raises(Exception):
            Message(
                role=MessageRole.USER,
                content="Hello",
                tool_calls=[ToolCall(name="read_file", arguments={})],
            )

    def test_tool_call_id_only_on_tool_result(self):
        msg = Message(
            role=MessageRole.TOOL_RESULT,
            content="File contents here",
            tool_call_id="call_123",
        )
        assert msg.tool_call_id == "call_123"

        with pytest.raises(Exception):
            Message(
                role=MessageRole.USER,
                content="Hello",
                tool_call_id="call_123",
            )


class TestToolCall:
    def test_valid_tool_call(self):
        tc = ToolCall(name="read_file", arguments={"path": "test.py"})
        assert tc.name == "read_file"

    def test_invalid_tool_name(self):
        with pytest.raises(Exception):
            ToolCall(name="ReadFile", arguments={})  # Not snake_case

    def test_empty_tool_name(self):
        with pytest.raises(Exception):
            ToolCall(name="", arguments={})


class TestExampleMetadata:
    def test_valid_metadata(self):
        meta = ExampleMetadata(
            category=Category.CODE_GENERATION,
            difficulty=Difficulty.MEDIUM,
            language="python",
            synthetic=True,
            allowed_for_training=True,
        )
        assert meta.category == Category.CODE_GENERATION

    def test_defaults(self):
        meta = ExampleMetadata(category=Category.DEBUGGING)
        assert meta.synthetic is False
        assert meta.environment == Environment.GENERAL
        assert meta.tags == []


class TestTrainingExample:
    def _make_example(self, **kwargs) -> dict:
        """Helper to create a valid example dict."""
        base = {
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Write a function."},
                {"role": "assistant", "content": "Here is the function: ..."},
            ],
            "metadata": {
                "category": "code_generation",
                "synthetic": True,
                "allowed_for_training": True,
            },
        }
        base.update(kwargs)
        return base

    def test_valid_example(self):
        data = self._make_example()
        example = TrainingExample.model_validate(data)
        assert len(example.messages) == 3

    def test_from_dict(self):
        data = self._make_example()
        example = TrainingExample.from_dict(data)
        assert example.metadata.category == Category.CODE_GENERATION

    def test_to_dict(self):
        data = self._make_example()
        example = TrainingExample.from_dict(data)
        result = example.to_dict()
        assert "messages" in result
        assert "metadata" in result

    def test_no_assistant_rejected(self):
        with pytest.raises(Exception):
            TrainingExample.model_validate({
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "metadata": {"category": "code_generation", "synthetic": True},
            })

    def test_first_message_must_be_user_or_system(self):
        with pytest.raises(Exception):
            TrainingExample.model_validate({
                "messages": [
                    {"role": "assistant", "content": "I start talking"},
                    {"role": "user", "content": "Ok"},
                ],
                "metadata": {"category": "code_generation", "synthetic": True},
            })

    def test_security_requires_authorization(self):
        with pytest.raises(Exception):
            TrainingExample.model_validate({
                "messages": [
                    {"role": "user", "content": "Review this code"},
                    {"role": "assistant", "content": "Found vulnerability"},
                ],
                "metadata": {
                    "category": "security_review",
                    # authorization defaults to NOT_APPLICABLE — should be rejected
                    "synthetic": True,
                },
            })

    def test_security_with_authorization_accepted(self):
        example = TrainingExample.model_validate({
            "messages": [
                {"role": "user", "content": "Review this code"},
                {"role": "assistant", "content": "Found vulnerability"},
            ],
            "metadata": {
                "category": "security_review",
                "authorization": "defensive",
                "synthetic": True,
            },
        })
        assert example.metadata.authorization == Authorization.DEFENSIVE

    def test_lab_requires_isolated_environment(self):
        with pytest.raises(Exception):
            TrainingExample.model_validate({
                "messages": [
                    {"role": "user", "content": "Exploit this"},
                    {"role": "assistant", "content": "In the lab environment..."},
                ],
                "metadata": {
                    "category": "authorized_lab",
                    "authorization": "authorized",
                    "environment": "general",  # should be isolated_lab, ctf, or educational
                    "synthetic": True,
                },
            })

    def test_lab_with_isolated_env_accepted(self):
        example = TrainingExample.model_validate({
            "messages": [
                {"role": "user", "content": "Test the lab app"},
                {"role": "assistant", "content": "Found SQL injection in lab"},
            ],
            "metadata": {
                "category": "authorized_lab",
                "authorization": "authorized",
                "environment": "isolated_lab",
                "synthetic": True,
            },
        })
        assert example.metadata.environment == Environment.ISOLATED_LAB

    def test_agent_trajectory(self):
        """Test multi-turn agent trajectory with tool calls."""
        example = TrainingExample.model_validate({
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Fix the bug."},
                {
                    "role": "assistant",
                    "content": "Let me read the file.",
                    "tool_calls": [
                        {"name": "read_file", "arguments": {"path": "app.py"}},
                    ],
                },
                {"role": "tool", "content": "file contents here"},
                {"role": "tool_result", "content": "success", "tool_call_id": "call_1"},
                {"role": "assistant", "content": "I found and fixed the bug."},
            ],
            "metadata": {
                "category": "agent_trajectories",
                "synthetic": True,
            },
        })
        assert len(example.messages) == 6


# ── Validation Tests ─────────────────────────────────────────


class TestValidation:
    def _write_jsonl(self, examples: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    def _valid_example(self) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "You are CyberCodeMini."},
                {"role": "user", "content": "Write hello world in Python."},
                {"role": "assistant", "content": "print('Hello, world!')"},
            ],
            "metadata": {
                "category": "code_generation",
                "difficulty": "easy",
                "language": "python",
                "source": "synthetic_dev",
                "synthetic": True,
                "allowed_for_training": True,
            },
        }

    def test_valid_file(self, tmp_path):
        from datasets.validators.validator import validate_file

        f = tmp_path / "valid.jsonl"
        self._write_jsonl([self._valid_example()], f)

        report = validate_file(str(f))
        assert report.total_examples == 1
        assert report.valid_examples == 1
        assert report.invalid_examples == 0
        assert not report.has_critical_errors

    def test_invalid_json(self, tmp_path):
        from datasets.validators.validator import validate_file

        f = tmp_path / "invalid.jsonl"
        f.write_text("this is not json\n")

        report = validate_file(str(f))
        assert report.invalid_examples == 1
        assert report.has_critical_errors

    def test_duplicate_detection(self, tmp_path):
        from datasets.validators.validator import validate_file

        ex = self._valid_example()
        f = tmp_path / "dupes.jsonl"
        self._write_jsonl([ex, ex], f)

        report = validate_file(str(f))
        assert report.duplicate_count == 1

    def test_secret_detection(self, tmp_path):
        from datasets.validators.validator import validate_file

        ex = self._valid_example()
        ex["messages"][2]["content"] = "Use api_key = 'sk-1234567890abcdefghij1234567890abcdef'"
        f = tmp_path / "secrets.jsonl"
        self._write_jsonl([ex], f)

        report = validate_file(str(f))
        assert report.potential_secrets >= 1

    def test_missing_provenance(self, tmp_path):
        from datasets.validators.validator import validate_file

        ex = self._valid_example()
        del ex["metadata"]["source"]
        ex["metadata"]["synthetic"] = False
        f = tmp_path / "no_provenance.jsonl"
        self._write_jsonl([ex], f)

        report = validate_file(str(f))
        assert report.missing_provenance >= 1

    def test_file_not_found(self):
        from datasets.validators.validator import validate_file

        report = validate_file("nonexistent.jsonl")
        assert report.has_critical_errors

    def test_report_to_dict(self, tmp_path):
        from datasets.validators.validator import validate_file

        f = tmp_path / "valid.jsonl"
        self._write_jsonl([self._valid_example()], f)

        report = validate_file(str(f))
        d = report.to_dict()
        assert "total_examples" in d
        assert "has_critical_errors" in d

    def test_dev_dataset_passes(self):
        """Ensure the actual dev dataset passes validation."""
        from datasets.validators.validator import validate_file

        dev_path = PROJECT_ROOT / "data" / "raw" / "dev_dataset.jsonl"
        if not dev_path.exists():
            pytest.skip("Dev dataset not found")

        report = validate_file(str(dev_path))
        assert report.total_examples >= 10
        assert not report.has_critical_errors, f"Dev dataset has errors:\n{report.summary()}"


# ── Deduplication Tests ──────────────────────────────────────


class TestDeduplication:
    def _valid_example(self, content: str = "Hello world") -> dict:
        return {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "Response"},
            ],
            "metadata": {"category": "code_generation", "synthetic": True},
        }

    def _write_jsonl(self, examples: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    def test_removes_exact_duplicates(self, tmp_path):
        from scripts.deduplicate import deduplicate

        ex = self._valid_example()
        input_f = tmp_path / "input.jsonl"
        output_f = tmp_path / "output.jsonl"
        self._write_jsonl([ex, ex, ex], input_f)

        stats = deduplicate(input_f, output_f, method="exact")
        assert stats["kept"] == 1
        assert stats["duplicates_removed"] == 2

    def test_keeps_unique(self, tmp_path):
        from scripts.deduplicate import deduplicate

        input_f = tmp_path / "input.jsonl"
        output_f = tmp_path / "output.jsonl"
        examples = [self._valid_example(f"Content {i}") for i in range(5)]
        self._write_jsonl(examples, input_f)

        stats = deduplicate(input_f, output_f, method="exact")
        assert stats["kept"] == 5
        assert stats["duplicates_removed"] == 0

    def test_normalized_dedup(self, tmp_path):
        from scripts.deduplicate import deduplicate

        ex1 = self._valid_example("Hello   World")
        ex2 = self._valid_example("hello   world")  # Different case
        input_f = tmp_path / "input.jsonl"
        output_f = tmp_path / "output.jsonl"
        self._write_jsonl([ex1, ex2], input_f)

        stats = deduplicate(input_f, output_f, method="normalized")
        assert stats["kept"] == 1
        assert stats["duplicates_removed"] == 1


# ── Split Tests ──────────────────────────────────────────────


class TestSplit:
    def _make_examples(self, n: int) -> list[dict]:
        examples = []
        sources = ["src_a", "src_b", "src_c", "src_d", "src_e"]
        for i in range(n):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
                "metadata": {
                    "category": "code_generation",
                    "source": sources[i % len(sources)],
                    "synthetic": True,
                },
            })
        return examples

    def _write_jsonl(self, examples: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    def test_random_split(self, tmp_path):
        from scripts.split_dataset import split_dataset

        input_f = tmp_path / "input.jsonl"
        self._write_jsonl(self._make_examples(20), input_f)

        stats = split_dataset(input_f, tmp_path)
        assert stats["train"] + stats["validation"] + stats["test"] == 20
        assert stats["train"] > 0
        assert stats["validation"] > 0
        assert stats["test"] > 0

    def test_group_split(self, tmp_path):
        from scripts.split_dataset import split_dataset

        input_f = tmp_path / "input.jsonl"
        self._write_jsonl(self._make_examples(20), input_f)

        stats = split_dataset(input_f, tmp_path, strategy="group", group_key="source")
        total = stats["train"] + stats["validation"] + stats["test"]
        assert total == 20

    def test_deterministic(self, tmp_path):
        from scripts.split_dataset import split_dataset

        input_f = tmp_path / "input.jsonl"
        self._write_jsonl(self._make_examples(20), input_f)

        stats1 = split_dataset(input_f, tmp_path, seed=42)
        stats2 = split_dataset(input_f, tmp_path, seed=42)
        assert stats1["train"] == stats2["train"]

    def test_empty_input_raises(self, tmp_path):
        from scripts.split_dataset import split_dataset

        input_f = tmp_path / "empty.jsonl"
        input_f.write_text("")

        with pytest.raises(ValueError, match="No examples"):
            split_dataset(input_f, tmp_path)


# ── Statistics Tests ─────────────────────────────────────────


class TestStatistics:
    def _write_jsonl(self, examples: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    def test_compute_stats(self, tmp_path):
        from scripts.dataset_stats import compute_stats

        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "metadata": {
                    "category": "code_generation",
                    "language": "python",
                    "difficulty": "easy",
                    "synthetic": True,
                    "source": "test",
                },
            },
            {
                "messages": [
                    {"role": "user", "content": "Review this"},
                    {"role": "assistant", "content": "Found issue"},
                ],
                "metadata": {
                    "category": "security_review",
                    "language": "javascript",
                    "difficulty": "medium",
                    "synthetic": False,
                    "source": "manual",
                },
            },
        ]
        f = tmp_path / "test.jsonl"
        self._write_jsonl(examples, f)

        stats = compute_stats(f)
        assert stats["total_examples"] == 2
        assert stats["categories"]["code_generation"] == 1
        assert stats["categories"]["security_review"] == 1
        assert stats["languages"]["python"] == 1
        assert stats["synthetic_count"] == 1
        assert stats["synthetic_percentage"] == 50.0

    def test_dev_dataset_stats(self):
        from scripts.dataset_stats import compute_stats

        dev_path = PROJECT_ROOT / "data" / "raw" / "dev_dataset.jsonl"
        if not dev_path.exists():
            pytest.skip("Dev dataset not found")

        stats = compute_stats(dev_path)
        assert stats["total_examples"] >= 10
        assert len(stats["categories"]) >= 4  # At least 4 categories represented


# ── Configuration Tests ──────────────────────────────────────


class TestConfiguration:
    def test_all_configs_load(self):
        import yaml

        configs = ["model.yaml", "dataset.yaml", "training.yaml", "evaluation.yaml", "azure.yaml"]
        for name in configs:
            path = PROJECT_ROOT / "configs" / name
            assert path.exists(), f"Config file missing: {name}"
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"Config file is empty: {name}"

    def test_azure_safety(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "azure.yaml") as f:
            config = yaml.safe_load(f)

        assert config["require_manual_confirmation"] is True, \
            "Azure must require manual confirmation"
        assert config["auto_shutdown"] is True, \
            "Azure must have auto_shutdown enabled"
        assert config["subscription_id"] == "", \
            "Azure subscription_id must not be hardcoded"

    def test_model_config_fields(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "model.yaml") as f:
            config = yaml.safe_load(f)

        required_fields = [
            "model_name", "model_revision", "tokenizer_name",
            "max_sequence_length", "trust_remote_code", "dtype",
        ]
        for field in required_fields:
            assert field in config, f"model.yaml missing field: {field}"

    def test_training_config_fields(self):
        import yaml

        with open(PROJECT_ROOT / "configs" / "training.yaml") as f:
            config = yaml.safe_load(f)

        required_fields = [
            "base_model", "epochs", "learning_rate", "batch_size",
            "lora_r", "lora_alpha", "lora_dropout", "seed",
        ]
        for field in required_fields:
            assert field in config, f"training.yaml missing field: {field}"
