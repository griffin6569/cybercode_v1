"""CyberCodeMini Phase 8 Comprehensive Unit Test Suite

Tests all tokenization, chat formatting, loss masking, tool trajectory,
collator, data leakage, and caching requirements for Phase 8.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from datasets.formatting.chat_formatter import ChatFormatter, convert_example_to_template_messages
from datasets.schemas.schema import Authorization, Category, Environment, MessageRole, ToolCall, TrainingExample
from datasets.tokenization import CyberCodeTokenizer, TokenizedExample, compute_tokenization_stats
from scripts.split_dataset import split_dataset
from training.collator import CyberCodeDataCollator


@pytest.fixture
def synthetic_agent_trajectory():
    return TrainingExample(
        messages=[
            {"role": MessageRole.SYSTEM, "content": "You are CyberCodeMini agent."},
            {"role": MessageRole.USER, "content": "Fix the security issue."},
            {
                "role": MessageRole.ASSISTANT,
                "content": "I will inspect the repository.",
                "tool_calls": [ToolCall(name="list_files", arguments={"path": "src/"})],
            },
            {
                "role": MessageRole.TOOL_RESULT,
                "content": "src/app.py\ntests/test_app.py",
                "tool_call_id": "call_01",
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "I will read app.py.",
                "tool_calls": [ToolCall(name="read_file", arguments={"path": "src/app.py"})],
            },
            {
                "role": MessageRole.TOOL_RESULT,
                "content": "query = f'SELECT * FROM u WHERE id={id}'",
                "tool_call_id": "call_02",
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "The input is unsafely concatenated into a SQL query. Fixing it now.",
                "tool_calls": [ToolCall(name="edit_file", arguments={"path": "src/app.py", "content": "query = 'SELECT * FROM u WHERE id=%s'"})],
            },
            {
                "role": MessageRole.TOOL_RESULT,
                "content": "File updated.",
                "tool_call_id": "call_03",
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "Running tests.",
                "tool_calls": [ToolCall(name="run_tests", arguments={"path": "tests/"})],
            },
            {
                "role": MessageRole.TOOL_RESULT,
                "content": "ALL TESTS PASSED",
                "tool_call_id": "call_04",
            },
            {
                "role": MessageRole.ASSISTANT,
                "content": "The vulnerability has been fixed and verified.",
            },
        ],
        metadata={
            "category": "agent_trajectories",
            "difficulty": "medium",
            "language": "python",
            "source": "synthetic_test",
            "source_id": "traj_synth_01",
            "authorization": "authorized",
            "environment": "isolated_lab",
            "synthetic": True,
            "allowed_for_training": True,
        },
    )


class TestChatFormatter:

    def test_convert_example_to_template_messages(self, synthetic_agent_trajectory):
        messages = convert_example_to_template_messages(synthetic_agent_trajectory)
        assert len(messages) == 11
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert len(messages[2]["tool_calls"]) == 1
        assert messages[2]["tool_calls"][0]["name"] == "list_files"
        assert messages[2]["tool_calls"][0]["arguments"] == {"path": "src/"}
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "call_01"

    def test_chatml_formatting(self, synthetic_agent_trajectory):
        formatter = ChatFormatter()
        messages = convert_example_to_template_messages(synthetic_agent_trajectory)
        formatted_str = formatter.format_chatml(messages)
        
        assert "<|im_start|>system\n" in formatted_str
        assert "<|im_start|>user\nFix the security issue." in formatted_str
        assert "<|im_start|>assistant\nI will inspect the repository." in formatted_str
        assert "list_files" in formatted_str
        assert "ALL TESTS PASSED" in formatted_str


class TestTokenizationAndLossMasking:

    def test_assistant_only_masking(self, synthetic_agent_trajectory):
        tokenizer = CyberCodeTokenizer(loss_masking_strategy="assistant_only", max_sequence_length=4096)
        tokenized = tokenizer.tokenize_example(synthetic_agent_trajectory)
        
        assert isinstance(tokenized, TokenizedExample)
        assert tokenized.example_id == "traj_synth_01"
        assert tokenized.final_token_count > 0
        # Check that labels contain -100 for masked turns
        assert -100 in tokenized.labels
        # Check that labels contain non-100 tokens for assistant turns
        assert any(l != -100 for l in tokenized.labels)

    def test_assistant_and_tool_outputs_masking(self, synthetic_agent_trajectory):
        tokenizer = CyberCodeTokenizer(loss_masking_strategy="assistant_and_tool_outputs", max_sequence_length=4096)
        tokenized = tokenizer.tokenize_example(synthetic_agent_trajectory)
        
        assert tokenized.final_token_count > 0
        assert -100 in tokenized.labels
        assert any(l != -100 for l in tokenized.labels)

    def test_all_messages_masking(self, synthetic_agent_trajectory):
        tokenizer = CyberCodeTokenizer(loss_masking_strategy="all_messages", max_sequence_length=4096)
        tokenized = tokenizer.tokenize_example(synthetic_agent_trajectory)
        
        # Under all_messages, no token should be masked (-100)
        assert -100 not in tokenized.labels

    def test_truncation_flagging(self, synthetic_agent_trajectory):
        # Set max sequence length very small to trigger truncation
        tokenizer = CyberCodeTokenizer(max_sequence_length=20)
        tokenized = tokenizer.tokenize_example(synthetic_agent_trajectory)
        
        assert tokenized.was_truncated is True
        assert tokenized.final_token_count == 20
        assert tokenized.original_token_count > 20

    def test_tokenization_stats_computation(self, synthetic_agent_trajectory):
        tokenizer = CyberCodeTokenizer(max_sequence_length=2048)
        tokenized = tokenizer.tokenize_example(synthetic_agent_trajectory)
        
        stats = compute_tokenization_stats([tokenized])
        assert stats["total_examples"] == 1
        assert stats["total_tokens"] == tokenized.final_token_count
        assert stats["mean_tokens"] == float(tokenized.final_token_count)
        assert "agent_trajectories" in stats["categories"]

    def test_cache_key_determinism(self):
        tokenizer = CyberCodeTokenizer(
            tokenizer_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            max_sequence_length=2048,
            loss_masking_strategy="assistant_and_tool_outputs",
        )
        key1 = tokenizer.compute_cache_key("hash_abc_123")
        key2 = tokenizer.compute_cache_key("hash_abc_123")
        key3 = tokenizer.compute_cache_key("hash_xyz_999")
        
        assert key1 == key2
        assert key1 != key3


class TestCollator:

    def test_data_collator_padding_and_masks(self):
        collator = CyberCodeDataCollator(pad_token_id=0, label_pad_token_id=-100, pad_to_multiple_of=8, max_length=2048)
        
        features = [
            {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1], "labels": [-100, 20, 30]},
            {"input_ids": [10, 20, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, -100, 30, 40, 50]},
        ]
        
        batch = collator(features)
        
        # Max length is 5 -> padded to multiple of 8 = 8
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch["attention_mask"]

        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
            labels = labels.tolist()
            attn_mask = attn_mask.tolist()

        assert len(input_ids) == 2
        assert len(input_ids[0]) == 8
        assert len(input_ids[1]) == 8
        
        # Verify padding values
        assert input_ids[0][3:] == [0, 0, 0, 0, 0]
        assert labels[0][3:] == [-100, -100, -100, -100, -100]
        assert attn_mask[0][3:] == [0, 0, 0, 0, 0]


from scripts.split_dataset import split_random

class TestDatasetLeakageProtection:

    def test_dataset_split_disjunction(self):
        examples = [
            {"messages": [{"role": "user", "content": f"Query #{i}"}, {"role": "assistant", "content": f"Response #{i}"}], "metadata": {"category": "code_generation", "source_id": f"ex_{i}"}}
            for i in range(30)
        ]
        
        train, val, test = split_random(examples, train_ratio=0.7, val_ratio=0.2, seed=42)
        
        train_ids = {ex["metadata"]["source_id"] for ex in train}
        val_ids = {ex["metadata"]["source_id"] for ex in val}
        test_ids = {ex["metadata"]["source_id"] for ex in test}

        # Verify pairwise empty intersections (no leakage)
        assert len(train_ids.intersection(val_ids)) == 0, "Train and Val IDs overlap!"
        assert len(train_ids.intersection(test_ids)) == 0, "Train and Test IDs overlap!"
        assert len(val_ids.intersection(test_ids)) == 0, "Val and Test IDs overlap!"
