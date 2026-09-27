"""CyberCodeMini Content Classifier & Metadata Manager

Classifies external dataset entries into domain, task_type, security_relevance,
authorization_context, content_type, and safety classifications.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ContentClassification:
    """Metadata classification for imported examples."""

    domain: str  # coding | debugging | security_review | vulnerability_remediation | agent_tool_use | authorized_lab | security_knowledge
    task_type: str  # coding | bug_localization | patch_generation | security_review | ctf | qa
    security_relevance: str  # "none" | "low" | "medium" | "high"
    authorization_context: str  # "not_applicable" | "explicit" | "authorized" | "defensive" | "educational" | "unknown"
    content_type: str  # "instruction" | "code" | "trajectory" | "qa" | "benchmark"
    safety_category: str  # "defensive" | "secure_coding" | "vulnerability_analysis" | "authorized_lab" | "security_benchmark" | "dual_use" | "offensive" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContentClassifier:
    """Rule-based content classifier for ingested dataset entries."""

    def classify(
        self,
        text: str,
        category_hint: Optional[str] = None,
        source_name: str = "",
    ) -> ContentClassification:
        text_lower = text.lower()
        src_lower = source_name.lower()

        # 1. Security Relevance & Category
        is_sec = any(kw in text_lower or kw in src_lower for kw in ["cve-", "cwe-", "vulnerability", "exploit", "sqli", "xss", "ssrf", "buffer overflow", "security", "secbench", "injection", "audit", "remediation", "patch"])
        is_agent = any(kw in text_lower or kw in src_lower for kw in ["tool_calls", "read_file", "edit_file", "run_tests", "swe-bench", "trajectory"])
        is_ctf = any(kw in text_lower or kw in src_lower for kw in ["ctf", "flag{", "wargame", "isolated_lab", "localhost"])

        if is_ctf:
            domain = "authorized_lab"
            task_type = "ctf"
            sec_rel = "high"
            auth_context = "explicit" if "localhost" in text_lower or "isolated_lab" in text_lower else "unknown"
            content_type = "instruction"
            safety = "authorized_lab"

        elif is_sec:
            if "patch" in text_lower or "remediation" in text_lower or "fix" in text_lower:
                domain = "vulnerability_remediation"
                task_type = "patch_generation"
                safety = "secure_coding"
            else:
                domain = "security_review"
                task_type = "security_review"
                safety = "vulnerability_analysis"

            sec_rel = "high"
            auth_context = "defensive"
            content_type = "code"

        elif is_agent:
            domain = "agent_tool_use"
            task_type = "bug_localization"
            sec_rel = "low"
            auth_context = "authorized"
            content_type = "trajectory"
            safety = "defensive"

        elif "debug" in text_lower or "error" in text_lower or "traceback" in text_lower:
            domain = "debugging"
            task_type = "bug_localization"
            sec_rel = "none"
            auth_context = "not_applicable"
            content_type = "code"
            safety = "defensive"

        else:
            domain = "coding"
            task_type = "coding"
            sec_rel = "none"
            auth_context = "not_applicable"
            content_type = "instruction"
            safety = "secure_coding"

        return ContentClassification(
            domain=domain,
            task_type=task_type,
            security_relevance=sec_rel,
            authorization_context=auth_context,
            content_type=content_type,
            safety_category=safety,
        )
