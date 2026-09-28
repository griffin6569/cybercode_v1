"""CyberCodeMini Dataset Schemas

Pydantic models defining the canonical dataset format for training examples,
including conversational messages, metadata, and agent tool-call trajectories.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MessageRole(str, Enum):
    """Allowed roles in a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    TOOL_RESULT = "tool_result"


class Category(str, Enum):
    """Dataset example categories."""

    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    SECURITY_REVIEW = "security_review"
    VULNERABILITY_REMEDIATION = "vulnerability_remediation"
    AGENT_TRAJECTORIES = "agent_trajectories"
    AUTHORIZED_LAB = "authorized_lab"
    CTF_CHALLENGE = "ctf_challenge"


class Difficulty(str, Enum):
    """Difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Environment(str, Enum):
    """Execution environment context."""

    EDUCATIONAL = "educational"
    ISOLATED_LAB = "isolated_lab"
    CTF = "ctf"
    PRODUCTION_REVIEW = "production_review"
    GENERAL = "general"


class Authorization(str, Enum):
    """Authorization status for security examples."""

    AUTHORIZED = "authorized"
    EXPLICIT = "explicit"
    DEFENSIVE = "defensive"
    EDUCATIONAL = "educational"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ToolCall(BaseModel):
    """Schema for agent tool calls within trajectories."""

    name: str = Field(..., min_length=1, max_length=100, description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    @field_validator("name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not re.match(r"^[a-z_][a-z0-9_]*$", v):
            raise ValueError(
                f"Tool name must be lowercase snake_case, got: '{v}'"
            )
        return v


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message content")
    tool_calls: Optional[list[ToolCall]] = Field(
        default=None, description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call this result responds to"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be empty or whitespace-only")
        return v

    @model_validator(mode="after")
    def validate_tool_fields(self) -> "Message":
        """Ensure tool_calls only on assistant messages, tool_call_id only on tool_result."""
        if self.tool_calls and self.role != MessageRole.ASSISTANT:
            raise ValueError("tool_calls can only be present on assistant messages")
        if self.tool_call_id and self.role != MessageRole.TOOL_RESULT:
            raise ValueError("tool_call_id can only be present on tool_result messages")
        return self


class ExampleMetadata(BaseModel):
    """Metadata attached to each training example."""

    category: Category = Field(..., description="Example category")
    difficulty: Optional[Difficulty] = Field(default=None, description="Difficulty level")
    environment: Optional[Environment] = Field(
        default=Environment.GENERAL, description="Execution environment"
    )
    authorization: Optional[Authorization] = Field(
        default=Authorization.NOT_APPLICABLE, description="Authorization context"
    )
    language: Optional[str] = Field(default=None, description="Primary programming language")
    vulnerability_type: Optional[str] = Field(
        default=None, description="Type of vulnerability (for security examples)"
    )
    cve_ids: list[str] = Field(default_factory=list, description="CVE identifiers")
    cwe_ids: list[str] = Field(default_factory=list, description="CWE identifiers")
    source: Optional[str] = Field(default=None, description="Source dataset or origin")
    source_id: Optional[str] = Field(default=None, description="Stable identifier in source dataset")
    source_url: Optional[str] = Field(default=None, description="URL of source data")
    license: Optional[str] = Field(default=None, description="License of the source data")
    author: Optional[str] = Field(default=None, description="Original author")
    original_id: Optional[str] = Field(default=None, description="ID in the source dataset")
    original_metadata: Optional[dict[str, Any]] = Field(default=None, description="Original raw metadata")
    collection_date: Optional[str] = Field(default=None, description="Date collected (ISO 8601)")
    synthetic: bool = Field(default=False, description="Whether the example is synthetic")
    transformed: bool = Field(default=False, description="Whether the example was transformed")
    allowed_for_training: Optional[bool] = Field(
        default=None,
        description="Whether training use is permitted. None means unknown.",
    )
    classification: Optional[str] = Field(
        default=None,
        description="Canonical classification: training_candidate, validation_candidate, evaluation_only, review, excluded",
    )
    source_details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Detailed provenance dictionary (dataset_id, revision, split, source_id, original_id)",
    )
    tags: list[str] = Field(default_factory=list, description="Additional tags")


class TrainingExample(BaseModel):
    """A single training example in CyberCodeMini's canonical format."""

    messages: list[Message] = Field(
        ..., min_length=2, description="Conversation messages (at least user + assistant)"
    )
    metadata: ExampleMetadata = Field(..., description="Example metadata")

    @model_validator(mode="after")
    def validate_message_sequence(self) -> "TrainingExample":
        """Validate that the message sequence is well-formed."""
        roles = [m.role for m in self.messages]

        # First message can be system, then must have user
        start_idx = 0
        if roles and roles[0] == MessageRole.SYSTEM:
            start_idx = 1

        if start_idx >= len(roles):
            raise ValueError("Messages must contain at least a user message after system")

        if roles[start_idx] != MessageRole.USER:
            raise ValueError(
                f"First non-system message must be 'user', got '{roles[start_idx].value}'"
            )

        # Must contain at least one assistant message
        if MessageRole.ASSISTANT not in roles:
            raise ValueError("Messages must contain at least one assistant message")

        # Security examples must have authorization
        if self.metadata.category in (
            Category.SECURITY_REVIEW,
            Category.VULNERABILITY_REMEDIATION,
            Category.AUTHORIZED_LAB,
        ):
            if self.metadata.authorization == Authorization.NOT_APPLICABLE:
                raise ValueError(
                    f"Security-related examples (category={self.metadata.category.value}) "
                    f"must specify an authorization context"
                )

        # Lab examples must have isolated environment
        if self.metadata.category == Category.AUTHORIZED_LAB:
            if self.metadata.environment not in (
                Environment.ISOLATED_LAB,
                Environment.CTF,
                Environment.EDUCATIONAL,
            ):
                raise ValueError(
                    f"Authorized lab examples must use isolated_lab, ctf, or educational "
                    f"environment, got '{self.metadata.environment}'"
                )

        return self

    def to_dict(self) -> dict:
        """Serialize to a dict suitable for JSONL output."""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        """Deserialize from a dict."""
        return cls.model_validate(data)


class DatasetRegistry(BaseModel):
    """Registry entry for an external dataset source."""

    dataset_name: str = Field(..., description="Name of the dataset")
    source: str = Field(..., description="Source (e.g., Hugging Face, GitHub)")
    source_url: Optional[str] = Field(default=None, description="URL")
    license: Optional[str] = Field(default=None, description="License identifier")
    training_allowed: Optional[bool] = Field(
        default=None, description="Whether training is permitted. None = unknown."
    )
    commercial_use: Optional[bool] = Field(
        default=None, description="Whether commercial use is permitted."
    )
    size: Optional[int] = Field(default=None, description="Approximate number of examples")
    domain: Optional[str] = Field(default=None, description="Domain (coding, security, etc.)")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    collection_date: Optional[str] = Field(default=None, description="Date added to registry")
