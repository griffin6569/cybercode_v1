"""CyberCodeMini Hugging Face Conversational Dataset Adapter

Converts Hugging Face format conversational datasets (ShareGPT, UltraFeedback,
OpenHermes, Instruction/Response pairs) into canonical CyberCodeMini schema.
"""

from __future__ import annotations

from typing import Any, Optional

from cybercode_datasets.adapters.base_adapter import BaseAdapter
from cybercode_datasets.provenance.provenance import DatasetRegistry
from cybercode_datasets.schemas.schema import (
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


class HuggingFaceConversationalAdapter(BaseAdapter):
    """Adapter for Hugging Face conversational / instruction datasets."""

    def __init__(
        self,
        source_name: str = "hf_conversational",
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        super().__init__(source_name=source_name, registry=registry, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a HF raw entry into a TrainingExample.

        Supports formats:
        - {"conversations": [{"from": "human"/"gpt", "value": "..."}]}
        - {"messages": [{"role": "user"/"assistant", "content": "..."}]}
        - {"instruction": "...", "response"/"output": "..."}
        """
        source_id = self._extract_source_id(raw_entry, default_prefix="hf")
        messages_list = []

        if "conversations" in raw_entry:
            # ShareGPT format
            for turn in raw_entry["conversations"]:
                speaker = str(turn.get("from", "")).lower()
                val = str(turn.get("value", "")).strip()
                if not val:
                    continue
                if speaker in ("human", "user"):
                    messages_list.append({"role": MessageRole.USER, "content": val})
                elif speaker in ("gpt", "assistant", "bot"):
                    messages_list.append({"role": MessageRole.ASSISTANT, "content": val})
                elif speaker == "system":
                    messages_list.append({"role": MessageRole.SYSTEM, "content": val})

        elif "messages" in raw_entry:
            # Standard OpenAI / HF messages format
            for msg in raw_entry["messages"]:
                role_str = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                try:
                    role = MessageRole(role_str)
                except ValueError:
                    if role_str in ("human", "user"):
                        role = MessageRole.USER
                    elif role_str in ("gpt", "assistant"):
                        role = MessageRole.ASSISTANT
                    elif role_str == "system":
                        role = MessageRole.SYSTEM
                    else:
                        continue
                messages_list.append({"role": role, "content": content})

        elif "instruction" in raw_entry or "prompt" in raw_entry:
            # Simple instruction / response pair
            instr = raw_entry.get("instruction") or raw_entry.get("prompt", "")
            resp = raw_entry.get("response") or raw_entry.get("output") or raw_entry.get("completion", "")
            
            instr = str(instr).strip()
            resp = str(resp).strip()

            if instr and resp:
                messages_list.append({"role": MessageRole.USER, "content": instr})
                messages_list.append({"role": MessageRole.ASSISTANT, "content": resp})

        if not messages_list:
            raise ValueError("Could not parse messages from raw entry")

        # Prepend default system prompt if missing
        if messages_list[0]["role"] != MessageRole.SYSTEM:
            messages_list.insert(0, {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT})

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

        return TrainingExample(messages=messages_list, metadata=metadata)
