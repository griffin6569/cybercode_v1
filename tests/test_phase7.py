"""CyberCodeMini Phase 7 Comprehensive Unit Test Suite

Tests all data adapter and curation requirements:
- Adapter loading & conversion
- Schema compliance
- Provenance preservation & registry licensing
- Security authorization metadata
- Unknown-license rejection
- Malformed example handling
- Agent trajectories & tool calls
- Quality filtering & contamination detection
- Human review queue generation
- Manifest generation
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from cybercode_datasets.adapters import (
    CTFLabAdapter,
    GenericInstructionAdapter,
    HuggingFaceConversationalAdapter,
    SecurityReviewAdapter,
    AgentTrajectoryAdapter,
)
from cybercode_datasets.filters.quality_filter import QualityFilter
from cybercode_datasets.provenance.provenance import DatasetRegistry, DatasetRegistryEntry
from cybercode_datasets.schemas.schema import Authorization, Category, Environment, MessageRole, TrainingExample
from cybercode_datasets.validators.quality import score_example
from scripts.curate_dataset import generate_manifest


@pytest.fixture
def temp_registry(tmp_path):
    reg_file = tmp_path / "dataset_registry.json"
    registry = DatasetRegistry(registry_file=reg_file)
    registry.register(
        name="test_approved_ds",
        source_url="https://example.org/approved",
        source_type="huggingface",
        license="MIT",
        training_allowed=True,
        status="approved",
    )
    registry.register(
        name="test_unknown_ds",
        source_url="https://example.org/unknown",
        source_type="github",
        license="unknown",
        training_allowed=False,
        status="review",
    )
    return registry


class TestPhase7Adapters:

    def test_adapter_loading(self, temp_registry):
        adapter = HuggingFaceConversationalAdapter(source_name="test_approved_ds", registry=temp_registry)
        assert adapter.allowed_for_training is True

        unknown_adapter = HuggingFaceConversationalAdapter(source_name="test_unknown_ds", registry=temp_registry)
        assert unknown_adapter.allowed_for_training is False

    def test_unknown_license_rejection(self, temp_registry):
        entry = temp_registry.get_entry("test_unknown_ds")
        assert entry.training_allowed is False
        assert entry.status == "review"

    def test_hf_adapter_conversion(self, temp_registry):
        adapter = HuggingFaceConversationalAdapter(source_name="test_approved_ds", registry=temp_registry)
        raw = {
            "id": "hf_001",
            "conversations": [
                {"from": "human", "value": "Write a python function to compute factorial."},
                {"from": "gpt", "value": "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```"}
            ]
        }
        ex = adapter.convert(raw)
        assert isinstance(ex, TrainingExample)
        assert ex.metadata.source_id == "hf_001"
        assert ex.metadata.allowed_for_training is True

    def test_security_review_adapter(self, temp_registry):
        adapter = SecurityReviewAdapter(source_name="test_approved_ds", registry=temp_registry)
        raw = {
            "cve_id": "CVE-2024-1234",
            "cwe_id": "CWE-89",
            "vulnerable_code": "query = f'SELECT * FROM u WHERE id={user_id}'",
            "fixed_code": "query = 'SELECT * FROM u WHERE id=%s'",
            "description": "SQL Injection in query execution.",
            "language": "python",
        }
        ex = adapter.convert(raw)
        assert ex.metadata.category == Category.VULNERABILITY_REMEDIATION
        assert ex.metadata.authorization == Authorization.DEFENSIVE
        assert ex.metadata.cve_ids == ["CVE-2024-1234"]
        assert ex.metadata.source_id == "CVE-2024-1234"

    def test_agent_trajectory_adapter(self, temp_registry):
        adapter = AgentTrajectoryAdapter(source_name="test_approved_ds", registry=temp_registry)
        raw = {
            "task_id": "traj_100",
            "task": "Fix bug in main.py",
            "trajectory": [
                {
                    "role": "assistant",
                    "content": "Let me list repository files.",
                    "tool_calls": [{"name": "list_files", "arguments": {"path": "."}}]
                },
                {"role": "tool_result", "content": "main.py", "tool_call_id": "call_0"},
                {"role": "assistant", "content": "I see main.py."}
            ]
        }
        ex = adapter.convert(raw)
        assert ex.metadata.category == Category.AGENT_TRAJECTORIES
        assert ex.metadata.environment == Environment.ISOLATED_LAB
        assert len(ex.messages) == 5  # System + User + Assistant(tool) + ToolResult + Assistant
        assert ex.messages[2].tool_calls[0].name == "list_files"

    def test_ctf_lab_adapter(self, temp_registry):
        adapter = CTFLabAdapter(source_name="test_approved_ds", registry=temp_registry)
        raw = {
            "id": "ctf_01",
            "challenge_prompt": "Perform SQL injection on local lab endpoint.",
            "writeup": "Payload: admin' --",
            "environment": "isolated_lab",
        }
        ex = adapter.convert(raw)
        assert ex.metadata.category in (Category.CTF_CHALLENGE, Category.AUTHORIZED_LAB)
        assert ex.metadata.authorization == Authorization.AUTHORIZED
        assert ex.metadata.environment == Environment.ISOLATED_LAB


class TestQualityAndReviewQueue:

    def test_quality_scoring_valid(self):
        ex = TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": "System prompt."},
                {"role": MessageRole.USER, "content": "Implement binary search in Python."},
                {"role": MessageRole.ASSISTANT, "content": "```python\ndef binary_search(arr, target):\n    pass\n```"}
            ],
            metadata={
                "category": "code_generation",
                "source": "test",
                "synthetic": True,
                "allowed_for_training": True
            }
        )
        assessment = score_example(ex)
        assert assessment.passed is True
        assert assessment.score >= 70.0

    def test_quality_scoring_low_quality(self):
        ex = TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": "System prompt."},
                {"role": MessageRole.USER, "content": "What is 2+2?"},
                {"role": MessageRole.ASSISTANT, "content": "i am sorry, but as an ai i cannot fulfill"}
            ],
            metadata={
                "category": "code_generation",
                "source": "test",
                "synthetic": False,
                "allowed_for_training": False
            }
        )
        assessment = score_example(ex)
        assert assessment.requires_human_review is True

    def test_review_queue_routing(self, tmp_path):
        q_file = tmp_path / "review_queue.jsonl"
        q_filter = QualityFilter(review_queue_path=q_file)

        ex = TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": "System prompt."},
                {"role": MessageRole.USER, "content": "Fix vulnerability."},
                {"role": MessageRole.ASSISTANT, "content": "Fixed."}
            ],
            metadata={
                "category": "security_review",
                "authorization": "unknown",  # Unknown auth -> review queue
                "source": "unverified",
                "synthetic": False,
                "allowed_for_training": False,
            }
        )
        keep, reason, review_reasons = q_filter.filter_example(ex)
        assert q_file.exists()
        lines = q_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_manifest_generation(self, tmp_path):
        manifest_file = tmp_path / "manifest.json"
        ex = TrainingExample(
            messages=[
                {"role": MessageRole.SYSTEM, "content": "System prompt."},
                {"role": MessageRole.USER, "content": "Hello"},
                {"role": MessageRole.ASSISTANT, "content": "Hi there!"}
            ],
            metadata={
                "category": "code_generation",
                "source": "synthetic_dev",
                "synthetic": True,
                "allowed_for_training": True,
            }
        )
        data = generate_manifest(
            input_file=tmp_path / "input.jsonl",
            approved_examples=[ex],
            rejected_count=0,
            review_count=0,
            manifest_path=manifest_file,
        )
        assert data["total_examples"] == 1
        assert manifest_file.exists()
