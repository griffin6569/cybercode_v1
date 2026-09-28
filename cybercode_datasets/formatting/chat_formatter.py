"""CyberCodeMini Chat Formatter

Applies tokenizer official chat template to canonical TrainingExample objects.
Preserves tool calls, arguments, tool results, and message order deterministically.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from cybercode_datasets.schemas.schema import MessageRole, TrainingExample


def convert_example_to_template_messages(example: TrainingExample) -> list[dict[str, Any]]:
    """Convert canonical TrainingExample messages into standard message dicts for tokenizer chat templates."""
    messages = []
    
    for msg in example.messages:
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        
        # Standardize role for chat template
        if role_str == MessageRole.TOOL_RESULT.value:
            role = "tool"
        else:
            role = role_str

        msg_dict: dict[str, Any] = {
            "role": role,
            "content": msg.content,
        }

        # Add tool_calls for assistant
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in msg.tool_calls
            ]

        # Add tool_call_id for tool results
        if msg.role == MessageRole.TOOL_RESULT and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        messages.append(msg_dict)

    return messages


class ChatFormatter:
    """Reusable chat formatter loading official model tokenizer chat template."""

    def __init__(self, tokenizer: Any = None, tokenizer_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> None:
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self._ensure_tokenizer()

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    trust_remote_code=True,
                )
            except Exception as err:
                print(f"Warning: Could not load tokenizer '{self.tokenizer_name}': {err}")

    def validate_chat_template(self) -> bool:
        """Validate that the loaded tokenizer has a valid chat template."""
        if self.tokenizer is None:
            return False
        return hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

    def format_example(self, example: TrainingExample, tokenize: bool = False) -> str | list[int]:
        """Format canonical TrainingExample using tokenizer chat template."""
        messages = convert_example_to_template_messages(example)

        if self.tokenizer is not None and self.validate_chat_template():
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=False,
                )
            except Exception as err:
                print(f"Warning: apply_chat_template failed: {err}, falling back to ChatML string format.")

        # Fallback ChatML formatter if tokenizer template fails or unavailable
        return self.format_chatml(messages)

    @staticmethod
    def format_chatml(messages: list[dict[str, Any]]) -> str:
        """Fallback ChatML formatting (<|im_start|>role\ncontent<|im_end|>)."""
        formatted_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls_str = json.dumps(msg["tool_calls"])
                content = f"{content}\n{tool_calls_str}".strip()

            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(formatted_parts) + "\n"
