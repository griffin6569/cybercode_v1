"""CyberCodeMini Tokenization, Loss Masking & Cache System

Transforms canonical TrainingExample datasets into tokenized PyTorch/Hugging Face format
containing input_ids, attention_mask, and labels with configurable loss masking strategies.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets.formatting.chat_formatter import ChatFormatter, convert_example_to_template_messages
from datasets.schemas.schema import MessageRole, TrainingExample


@dataclass
class TokenizedExample:
    """A tokenized training example with loss-masked labels and metadata."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]  # -100 for masked tokens
    example_id: str
    category: str
    source: str
    difficulty: str
    was_truncated: bool = False
    original_token_count: int = 0
    final_token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CyberCodeTokenizer:
    """Tokenizer and loss masking engine for CyberCodeMini."""

    def __init__(
        self,
        tokenizer: Any = None,
        tokenizer_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_sequence_length: int = 2048,
        loss_masking_strategy: str = "assistant_and_tool_outputs",  # assistant_only | assistant_and_tool_outputs | all_messages
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = max_sequence_length
        self.loss_masking_strategy = loss_masking_strategy
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/interim")
        
        self.formatter = ChatFormatter(tokenizer=self.tokenizer, tokenizer_name=self.tokenizer_name)
        self.tokenizer = self.formatter.tokenizer

    def _get_message_spans(self, example: TrainingExample) -> list[dict[str, Any]]:
        """Tokenize message by message to compute token role boundaries for loss masking."""
        spans = []
        
        if self.tokenizer is None:
            # Simple mock tokenization fallback for testing without HF models
            for msg in example.messages:
                content = msg.content
                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                tokens = [ord(c) % 100 + 10 for c in content]  # Dummy token ids
                spans.append({
                    "role": role_str,
                    "tokens": tokens,
                    "has_tool_calls": bool(msg.tool_calls),
                })
            return spans

        # Real tokenizer path
        for msg in example.messages:
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            
            # Format message turn
            formatted_turn = self.formatter.format_chatml([{
                "role": role_str if role_str != MessageRole.TOOL_RESULT.value else "tool",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
            }])

            tokens = self.tokenizer.encode(formatted_turn, add_special_tokens=False)
            spans.append({
                "role": role_str,
                "tokens": tokens,
                "has_tool_calls": bool(msg.tool_calls),
            })

        return spans

    def tokenize_example(self, example: TrainingExample) -> TokenizedExample:
        """Tokenize single TrainingExample and construct loss-masked labels."""
        spans = self._get_message_spans(example)
        
        input_ids: list[int] = []
        labels: list[int] = []

        for span in spans:
            role = span["role"]
            tokens = span["tokens"]

            input_ids.extend(tokens)

            # Determine whether to compute loss on this turn based on strategy
            should_mask = True

            if self.loss_masking_strategy == "all_messages":
                should_mask = False
            elif self.loss_masking_strategy == "assistant_only":
                if role == MessageRole.ASSISTANT.value:
                    should_mask = False
            elif self.loss_masking_strategy == "assistant_and_tool_outputs":
                if role in (MessageRole.ASSISTANT.value, MessageRole.TOOL.value):
                    should_mask = False

            if should_mask:
                labels.extend([-100] * len(tokens))
            else:
                labels.extend(tokens)

        original_count = len(input_ids)
        was_truncated = False

        if original_count > self.max_sequence_length:
            input_ids = input_ids[: self.max_sequence_length]
            labels = labels[: self.max_sequence_length]
            was_truncated = True

        final_count = len(input_ids)
        attention_mask = [1] * final_count

        meta = example.metadata
        ex_id = meta.source_id or meta.original_id or "ex_000"
        cat = meta.category.value if hasattr(meta.category, "value") else str(meta.category)
        diff = meta.difficulty.value if hasattr(meta.difficulty, "value") else str(meta.difficulty)

        return TokenizedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            example_id=ex_id,
            category=cat,
            source=meta.source or "unknown",
            difficulty=diff,
            was_truncated=was_truncated,
            original_token_count=original_count,
            final_token_count=final_count,
        )

    def tokenize_dataset(self, examples: Sequence[TrainingExample]) -> list[TokenizedExample]:
        """Tokenize a full sequence of TrainingExamples."""
        return [self.tokenize_example(ex) for ex in examples]

    def compute_cache_key(self, dataset_hash: str) -> str:
        """Compute deterministic cache key based on configuration parameters."""
        raw_key = f"{dataset_hash}_{self.tokenizer_name}_{self.max_sequence_length}_{self.loss_masking_strategy}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def compute_tokenization_stats(tokenized_examples: Sequence[TokenizedExample]) -> dict[str, Any]:
    """Compute tokenization statistics broken down overall and per category."""
    if not tokenized_examples:
        return {}

    lengths = [ex.final_token_count for ex in tokenized_examples]
    truncated = sum(1 for ex in tokenized_examples if ex.was_truncated)
    total = len(tokenized_examples)

    sorted_lengths = sorted(lengths)
    p95_idx = min(int(math.ceil(0.95 * total)) - 1, total - 1)
    p95_val = sorted_lengths[p95_idx] if total > 0 else 0

    stats: dict[str, Any] = {
        "total_examples": total,
        "total_tokens": sum(lengths),
        "mean_tokens": round(statistics.mean(lengths), 2) if total else 0,
        "median_tokens": round(statistics.median(lengths), 2) if total else 0,
        "p95_tokens": p95_val,
        "maximum_tokens": max(lengths) if total else 0,
        "truncated_examples": truncated,
        "truncation_percentage": round((truncated / total) * 100, 2) if total else 0.0,
        "categories": {},
    }

    # Group by category
    by_category: dict[str, list[int]] = {}
    for ex in tokenized_examples:
        by_category.setdefault(ex.category, []).append(ex.final_token_count)

    for cat, cat_lens in by_category.items():
        c_tot = len(cat_lens)
        stats["categories"][cat] = {
            "count": c_tot,
            "total_tokens": sum(cat_lens),
            "mean_tokens": round(statistics.mean(cat_lens), 2),
            "median_tokens": round(statistics.median(cat_lens), 2),
            "maximum_tokens": max(cat_lens),
        }

    return stats
