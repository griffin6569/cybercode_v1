"""CyberCodeMini Data Quality Scoring & Assessment

Scores training examples on a 0-100 scale evaluating:
- Completeness
- Clarity
- Provenance & License authorization
- Metadata completeness
- Response usefulness / substance
- Tool-call validity
- Security-context completeness & explicit authorization
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets.schemas.schema import Authorization, MessageRole, TrainingExample

# Common low-information / useless phrases to penalize
LOW_INFO_RESPONSES = [
    r"as an ai",
    r"i cannot fulfill",
    r"i am sorry, but",
    r"i don't know",
    r"tbd",
    r"todo",
    r"placeholder",
    r"hello world",
]


@dataclass
class QualityAssessment:
    """Detailed quality assessment for a training example."""

    score: float  # 0.0 - 100.0
    passed: bool
    reasons: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "passed": self.passed,
            "reasons": self.reasons,
            "requires_human_review": self.requires_human_review,
            "review_reasons": self.review_reasons,
        }


def score_example(example: TrainingExample) -> QualityAssessment:
    """Evaluate and score a TrainingExample on a 0-100 scale.
    
    Returns a QualityAssessment detailing score breakdown and human review triggers.
    """
    score = 100.0
    reasons = []
    review_reasons = []

    meta = example.metadata
    messages = example.messages

    # 1. Message Structure & Completeness (Max 25 pts)
    if len(messages) < 2:
        score -= 25.0
        reasons.append("Conversation has fewer than 2 messages")

    has_user = any(m.role == MessageRole.USER for m in messages)
    has_assistant = any(m.role == MessageRole.ASSISTANT for m in messages)

    if not has_user:
        score -= 15.0
        reasons.append("Missing user query")
    if not has_assistant:
        score -= 20.0
        reasons.append("Missing assistant response")

    # Check for empty assistant responses
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.ASSISTANT:
            content = msg.content.strip()
            if not content:
                score -= 30.0
                reasons.append(f"Empty assistant response at index {i}")
                review_reasons.append("Empty assistant response")
            elif len(content) < 15:
                score -= 10.0
                reasons.append(f"Very short assistant response ({len(content)} chars)")

            # Check low-information responses
            for pattern in LOW_INFO_RESPONSES:
                if re.search(pattern, content, re.IGNORECASE):
                    score -= 15.0
                    reasons.append(f"Low-information response matching '{pattern}'")
                    review_reasons.append(f"Low-information phrase detected: {pattern}")
                    break

    # 2. Tool-Call Validity (Max 20 pts)
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                if not tc.name:
                    score -= 15.0
                    reasons.append("Malformed tool call: missing name")
                    review_reasons.append("Malformed tool call")
                if tc.name not in ("read_file", "edit_file", "run_tests", "list_files", "search_code", "execute_command"):
                    # Unrecognized tool name penalty
                    score -= 5.0
                    reasons.append(f"Non-standard tool call name: '{tc.name}'")

    # 3. Security Context & Explicit Authorization (Max 25 pts)
    if meta.category in ("security_review", "vulnerability_remediation", "authorized_lab", "ctf_challenge"):
        # Check authorization string
        auth_str = str(meta.authorization.value if hasattr(meta.authorization, "value") else meta.authorization).lower()
        if auth_str in ("unknown", "unclear", "none"):
            score -= 20.0
            reasons.append("Security example missing explicit authorization")
            review_reasons.append("Unknown or unverified authorization")
        
        # Check environment
        env_str = str(meta.environment.value if hasattr(meta.environment, "value") else meta.environment).lower()
        if env_str not in ("isolated_lab", "ctf", "educational"):
            score -= 10.0
            reasons.append(f"Security task in non-isolated environment: '{env_str}'")

    # 4. Provenance & Metadata Completeness (Max 15 pts)
    if not meta.source and not meta.synthetic:
        score -= 15.0
        reasons.append("Non-synthetic example missing source provenance")
        review_reasons.append("Missing provenance")

    if meta.allowed_for_training is False and not meta.synthetic:
        score -= 20.0
        reasons.append("Dataset license does not permit training")
        review_reasons.append("Unknown or unallowed training license")

    if not meta.source_id:
        score -= 5.0
        reasons.append("Missing source_id")

    # 5. Finalize Score & Review Triggers
    final_score = max(0.0, score)
    passed = final_score >= 70.0

    requires_review = (
        final_score < 70.0
        or len(review_reasons) > 0
        or meta.authorization == Authorization.UNKNOWN
        or (meta.allowed_for_training is False and not meta.synthetic)
    )

    return QualityAssessment(
        score=final_score,
        passed=passed,
        reasons=reasons,
        requires_human_review=requires_review,
        review_reasons=review_reasons,
    )
