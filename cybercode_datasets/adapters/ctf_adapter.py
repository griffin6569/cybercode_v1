"""CyberCodeMini CTF Challenge Dataset Adapter

Converts CTF challenges and writeups into canonical TrainingExample dialogues
with mandatory environment & authorization tags.
"""

from __future__ import annotations

from typing import Any

from cybercode_datasets.adapters.base_adapter import BaseAdapter
from cybercode_datasets.schemas.schema import (
    Authorization,
    Category,
    Difficulty,
    Environment,
    ExampleMetadata,
    MessageRole,
    TrainingExample,
)

SYSTEM_PROMPT = (
    "You are CyberCodeMini, a coding and cybersecurity engineering assistant "
    "specialized in secure software development, vulnerability analysis, debugging, "
    "and authorized security testing."
)


class CTFAdapter(BaseAdapter):
    """Adapter for CTF challenges and lab writeup datasets."""

    def __init__(self, source_name: str = "ctf_challenges", default_synthetic: bool = False) -> None:
        super().__init__(source_name=source_name, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a CTF raw entry into a TrainingExample dialogue.

        Expected keys in raw_entry:
            - challenge_prompt (str): The CTF prompt / problem statement
            - writeup (str): Solution / analysis writeup
            - category_name (str, optional): e.g. web, pwn, reverse, crypto
            - difficulty (str, optional): easy / medium / hard
            - environment (str, optional): isolated_lab / ctf
        """
        prompt = raw_entry.get("challenge_prompt", "").strip()
        writeup = raw_entry.get("writeup", "").strip()

        if not prompt or not writeup:
            raise ValueError("CTF entry must contain both 'challenge_prompt' and 'writeup'")

        lang = raw_entry.get("language", "python").lower()
        diff_str = raw_entry.get("difficulty", "medium").lower()

        try:
            difficulty = Difficulty(diff_str)
        except ValueError:
            difficulty = Difficulty.MEDIUM

        messages = [
            {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
            {"role": MessageRole.USER, "content": prompt},
            {"role": MessageRole.ASSISTANT, "content": writeup},
        ]

        metadata = ExampleMetadata(
            category=Category.CTF_CHALLENGE,
            difficulty=difficulty,
            language=lang,
            source=self.source_name,
            authorization=Authorization.AUTHORIZED,
            environment=Environment.CTF,
            synthetic=self.default_synthetic,
            allowed_for_training=True,
        )

        return TrainingExample(messages=messages, metadata=metadata)
