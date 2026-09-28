"""CyberCodeMini CTF & Authorized Lab Dataset Adapter

Converts CTF challenges, wargame prompts, and security lab writeups into
canonical authorized-lab schema instances.
"""

from __future__ import annotations

from typing import Any, Optional

from cybercode_datasets.adapters.base_adapter import BaseAdapter
from cybercode_datasets.provenance.provenance import DatasetRegistry
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
    "and authorized security testing in isolated environments."
)


class CTFLabAdapter(BaseAdapter):
    """Adapter for CTF challenges and lab task writeup datasets."""

    def __init__(
        self,
        source_name: str = "ctf_lab_db",
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        super().__init__(source_name=source_name, registry=registry, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a CTF / lab challenge entry into a TrainingExample.

        Expected fields:
            - challenge_prompt / task (str)
            - writeup / solution (str)
            - category_name (str, optional: web, pwn, reverse, crypto, forensics)
            - environment (str, optional: isolated_lab, ctf)
        """
        source_id = self._extract_source_id(raw_entry, default_prefix="ctf")

        prompt = raw_entry.get("challenge_prompt") or raw_entry.get("task") or raw_entry.get("problem") or ""
        writeup = raw_entry.get("writeup") or raw_entry.get("solution") or raw_entry.get("flag_path") or ""

        prompt = str(prompt).strip()
        writeup = str(writeup).strip()

        if not prompt or not writeup:
            raise ValueError("CTF/Lab entry must contain both challenge prompt and solution/writeup")

        env_str = str(raw_entry.get("environment", "isolated_lab")).lower()
        try:
            env = Environment(env_str)
        except ValueError:
            env = Environment.ISOLATED_LAB

        cat_name = str(raw_entry.get("category_name", "")).lower()
        if "ctf" in cat_name or "wargame" in self.source_name.lower():
            category = Category.CTF_CHALLENGE
        else:
            category = Category.AUTHORIZED_LAB

        lang = str(raw_entry.get("language", "python")).lower()
        diff_str = str(raw_entry.get("difficulty", "medium")).lower()

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
            category=category,
            difficulty=difficulty,
            language=lang,
            source=self.source_name,
            source_id=source_id,
            authorization=Authorization.AUTHORIZED,
            environment=env,
            synthetic=raw_entry.get("synthetic", self.default_synthetic),
            allowed_for_training=self.allowed_for_training,
        )

        return TrainingExample(messages=messages, metadata=metadata)
