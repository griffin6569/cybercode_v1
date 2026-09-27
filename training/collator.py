"""CyberCodeMini Data Collator for Conversational Fine-Tuning

Pads variable-length tokenized input_ids, attention_mask, and labels to max length
in a batch. Uses label padding value -100 to ignore padded positions in PyTorch loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class CyberCodeDataCollator:
    """Data collator for conversational supervised fine-tuning."""

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8
    max_length: Optional[int] = 2048
    packing: bool = False  # Sequence packing disabled by default

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of tokenized feature dicts into padded PyTorch Tensors (or nested lists)."""
        if not features:
            return {}

        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        batch_labels = [f.get("labels", f["input_ids"]) for f in features]

        # Calculate max length in batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        if self.max_length is not None:
            max_batch_len = min(max_batch_len, self.max_length)

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = max_batch_len % self.pad_to_multiple_of
            if remainder != 0:
                max_batch_len += self.pad_to_multiple_of - remainder

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attn_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            # Truncate if exceeds max_batch_len
            input_ids = input_ids[:max_batch_len]
            attn_mask = attn_mask[:max_batch_len]
            labels = labels[:max_batch_len]

            pad_len = max_batch_len - len(input_ids)

            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)
            padded_labels.append(labels + [self.label_pad_token_id] * pad_len)

        # Attempt to convert to PyTorch tensors if PyTorch is available
        try:
            import torch
            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
            }
        except ImportError:
            # Fallback to lists if PyTorch not installed
            return {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels,
            }
