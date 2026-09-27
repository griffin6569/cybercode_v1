"""CyberCodeMini Agent Trajectory Dataset Adapter

Converts coding-agent tool execution logs into canonical multi-turn tool-call trajectory format.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from datasets.adapters.base_adapter import BaseAdapter
from datasets.provenance.provenance import DatasetRegistry
from datasets.schemas.schema import (
    Authorization,
    Category,
    Difficulty,
    Environment,
    ExampleMetadata,
    MessageRole,
    ToolCall,
    TrainingExample,
)

SYSTEM_PROMPT = (
    "You are CyberCodeMini, an agentic coding and cybersecurity engineering model. "
    "You solve software engineering tasks using sandboxed tool calls."
)


class AgentTrajectoryAdapter(BaseAdapter):
    """Adapter for coding-agent trajectory and tool-use datasets."""

    def __init__(
        self,
        source_name: str = "agent_trajectories",
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        super().__init__(source_name=source_name, registry=registry, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a raw trajectory entry into a TrainingExample.

        Expected fields:
            - trajectory / turns (list of dicts representing steps)
            - task / user_request (str)
        """
        source_id = self._extract_source_id(raw_entry, default_prefix="traj")
        task_prompt = raw_entry.get("task") or raw_entry.get("user_request") or raw_entry.get("instruction", "")
        task_prompt = str(task_prompt).strip()

        if not task_prompt:
            raise ValueError("Trajectory entry missing 'task' or 'user_request'")

        messages = [
            {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
            {"role": MessageRole.USER, "content": task_prompt},
        ]

        turns = raw_entry.get("trajectory") or raw_entry.get("turns") or raw_entry.get("steps") or []
        
        for turn in turns:
            role_str = str(turn.get("role", "")).lower()
            content = str(turn.get("content", "")).strip()

            if role_str == "assistant":
                msg_dict: dict[str, Any] = {"role": MessageRole.ASSISTANT, "content": content or "Executing tool."}
                
                # Check for tool_calls
                if "tool_calls" in turn and turn["tool_calls"]:
                    tool_calls = []
                    for tc in turn["tool_calls"]:
                        name = str(tc.get("name", ""))
                        args = tc.get("arguments") or tc.get("args") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"raw": args}
                        tool_calls.append(ToolCall(name=name, arguments=args))
                    msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)

            elif role_str in ("tool", "tool_result"):
                call_id = str(turn.get("tool_call_id") or turn.get("call_id") or "call_0")
                messages.append({
                    "role": MessageRole.TOOL_RESULT,
                    "content": content or "success",
                    "tool_call_id": call_id,
                })

            elif role_str == "user":
                if content:
                    messages.append({"role": MessageRole.USER, "content": content})

        lang = str(raw_entry.get("language", "python")).lower()
        diff_str = str(raw_entry.get("difficulty", "medium")).lower()

        try:
            difficulty = Difficulty(diff_str)
        except ValueError:
            difficulty = Difficulty.MEDIUM

        metadata = ExampleMetadata(
            category=Category.AGENT_TRAJECTORIES,
            difficulty=difficulty,
            language=lang,
            source=self.source_name,
            source_id=source_id,
            authorization=Authorization.AUTHORIZED,
            environment=Environment.ISOLATED_LAB,
            synthetic=raw_entry.get("synthetic", self.default_synthetic),
            allowed_for_training=self.allowed_for_training,
        )

        return TrainingExample(messages=messages, metadata=metadata)
