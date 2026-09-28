"""CyberCodeMini Security Review & Vulnerability Remediation Dataset Adapter

Converts CVE/CWE databases, patch commits, and security review data into
security-review and remediation dialogue format with security-context metadata.
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
    "and authorized security testing."
)


class SecurityReviewAdapter(BaseAdapter):
    """Adapter for security review, CVE analysis, and vulnerability remediation datasets."""

    def __init__(
        self,
        source_name: str = "security_review_db",
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        super().__init__(source_name=source_name, registry=registry, default_synthetic=default_synthetic)

    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a security review / CVE record into a TrainingExample.

        Expected fields:
            - vulnerable_code / code (str)
            - fixed_code / patch (str, optional)
            - description / advisory (str, optional)
            - cve_id / cve (str, optional)
            - cwe_id / cwe (str, optional)
            - language (str, optional)
        """
        source_id = self._extract_source_id(raw_entry, default_prefix="sec")
        
        vuln_code = (
            raw_entry.get("vulnerable_code")
            or raw_entry.get("code")
            or raw_entry.get("func_before")
            or ""
        )
        vuln_code = str(vuln_code).strip()

        if not vuln_code:
            raise ValueError("Raw entry missing vulnerable code snippet")

        fixed_code = (
            raw_entry.get("fixed_code")
            or raw_entry.get("patch")
            or raw_entry.get("func_after")
            or ""
        )
        fixed_code = str(fixed_code).strip()

        description = (
            raw_entry.get("description")
            or raw_entry.get("advisory")
            or raw_entry.get("commit_message")
            or "Review code for vulnerabilities and provide remediation."
        )
        description = str(description).strip()

        cve_id = str(raw_entry.get("cve_id") or raw_entry.get("cve") or "").strip()
        cwe_id = str(raw_entry.get("cwe_id") or raw_entry.get("cwe") or "").strip()
        lang = str(raw_entry.get("language", "python")).lower()
        diff_str = str(raw_entry.get("difficulty", "medium")).lower()

        try:
            difficulty = Difficulty(diff_str)
        except ValueError:
            difficulty = Difficulty.MEDIUM

        # Format user prompt
        user_content = (
            f"Perform a security review of the following {lang.capitalize()} code snippet.\n"
            f"Identify any security flaws (CWE/CVE) and provide a secure remediation:\n\n"
            f"```{lang}\n{vuln_code}\n```"
        )

        # Format assistant response
        header = f"## Security Review"
        if cve_id or cwe_id:
            header += f" ({' / '.join(filter(None, [cve_id, cwe_id]))})"

        response_parts = [header, "", description]
        if fixed_code:
            response_parts.extend(["", "### Remediation Patch", "", f"```{lang}\n{fixed_code}\n```"])

        assistant_content = "\n".join(response_parts)

        messages = [
            {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT},
            {"role": MessageRole.USER, "content": user_content},
            {"role": MessageRole.ASSISTANT, "content": assistant_content},
        ]

        cve_list = [cve_id] if cve_id else []
        cwe_list = [cwe_id] if cwe_id else []

        metadata = ExampleMetadata(
            category=Category.VULNERABILITY_REMEDIATION,
            difficulty=difficulty,
            language=lang,
            source=self.source_name,
            source_id=source_id,
            cve_ids=cve_list,
            cwe_ids=cwe_list,
            authorization=Authorization.DEFENSIVE,
            environment=Environment.EDUCATIONAL,
            synthetic=raw_entry.get("synthetic", self.default_synthetic),
            allowed_for_training=self.allowed_for_training,
        )

        return TrainingExample(messages=messages, metadata=metadata)
