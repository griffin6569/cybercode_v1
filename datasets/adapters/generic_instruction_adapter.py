"""CyberCodeMini Generic Instruction Dataset Adapter

Converts generic instruction/chat JSONL entries into canonical TrainingExample schema.
"""

from __future__ import annotations

from typing import Any, Optional

from datasets.adapters.base_adapter import DatasetAdapter
from datasets.provenance.provenance import DatasetRegistry
from datasets.schemas.schema import (
    Authorization,
    Category,
    Difficulty,
    ExampleMetadata,
    MessageRole,
    TrainingExample,
)

SYSTEM_PROMPT = (
    "You are CyberCodeMini, a coding and cybersecurity engineering assistant "
    "specialized in secure software development, vulnerability analysis, debugging, "
    "and authorized security testing."
)


class GenericInstructionAdapter(DatasetAdapter):
    """Adapter for generic instruction/response JSONL datasets."""

    def __init__(
        self,
        source_name: str = "generic_instruction",
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        super().__init__(source_name=source_name, registry=registry, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert generic instruction raw dict into TrainingExample."""
        source_id = self._extract_source_id(raw_entry, default_prefix="instr")
        
        prompt = raw_entry.get("instruction") or raw_entry.get("prompt") or raw_entry.get("input", "")
        response = raw_entry.get("output") or raw_entry.get("response") or raw_entry.get("target", "")

        prompt = str(prompt).strip()
        response = str(response).strip()

        if not prompt or not response:
            raise ValueError("Generic instruction entry must have both prompt/instruction and response/output")

        messages = [
            {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
            {"role": MessageRole.USER, "content": prompt},
            {"role": MessageRole.ASSISTANT, "content": response},
        ]

        lang = str(raw_entry.get("language", "python")).lower()
        diff_str = str(raw_entry.get("difficulty", "medium")).lower()

        try:
            difficulty = Difficulty(diff_str)
        except ValueError:
            difficulty = Difficulty.MEDIUM

        metadata = ExampleMetadata(
            category=Category.CODE_GENERATION,
            difficulty=difficulty,
            language=lang,
            source=self.source_name,
            source_id=source_id,
            authorization=Authorization.DEFENSIVE,
            synthetic=raw_entry.get("synthetic", self.default_synthetic),
            allowed_for_training=self.allowed_for_training,
        )

        return TrainingExample(messages=messages, metadata=metadata)
