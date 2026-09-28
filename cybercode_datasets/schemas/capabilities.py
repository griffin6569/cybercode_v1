"""CyberCodeMini Capability Taxonomy

Defines the canonical taxonomy for model capability classification, target distributions,
and multi-label capability assignment across training and evaluation datasets.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Capability(str, Enum):
    """Canonical capability taxonomy for CyberCodeMini."""

    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    SECURE_CODING = "secure_coding"
    SECURITY_REVIEW = "security_review"
    VULNERABILITY_DETECTION = "vulnerability_detection"
    VULNERABILITY_REMEDIATION = "vulnerability_remediation"
    SECURITY_REASONING = "security_reasoning"
    TEST_GENERATION = "test_generation"
    REPOSITORY_NAVIGATION = "repository_navigation"
    SOFTWARE_ENGINEERING = "software_engineering"
    REFACTORING = "refactoring"
    AGENT_TOOL_USE = "agent_tool_use"
    AGENT_PLANNING = "agent_planning"
    AGENT_EXECUTION = "agent_execution"
    AGENT_TRAJECTORIES = "agent_trajectories"
    AUTHORIZED_LAB = "authorized_lab"
    CTF = "ctf"
    DOCUMENTATION = "documentation"
    CODE_EXPLANATION = "code_explanation"


class CapabilityClassification(BaseModel):
    """Capability classification structure attached to metadata."""

    primary_capability: Capability = Field(..., description="Primary capability demonstrated by example")
    secondary_capabilities: List[Capability] = Field(
        default_factory=list, description="Secondary or supporting capabilities"
    )


# Target capability percentage distribution ranges for training corpus
TARGET_CAPABILITY_RANGES: dict[str, tuple[float, float]] = {
    "code_generation": (15.0, 20.0),
    "debugging": (10.0, 15.0),
    "security_review": (10.0, 15.0),
    "vulnerability_detection": (5.0, 10.0),
    "vulnerability_remediation": (15.0, 20.0),
    "security_reasoning": (5.0, 10.0),
    "agent_tool_use": (10.0, 15.0),
    "software_engineering": (5.0, 10.0),
    "test_generation": (5.0, 10.0),
    "authorized_lab": (5.0, 10.0),
    "refactoring": (5.0, 10.0),
}
