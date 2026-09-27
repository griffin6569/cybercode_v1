"""CyberCodeMini Dataset Validators

Validates JSONL dataset files against the canonical schema, checking for:
- Valid JSON structure
- Required fields and valid message roles
- Non-empty content
- Reasonable message lengths
- Duplicate examples
- Malformed tool calls
- Missing metadata
- Potential secrets/PII
- Invalid Unicode
- Excessively long examples
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.schemas.schema import MessageRole, TrainingExample


# Common dummy/placeholder secret strings used in educational code/prompts
DUMMY_SECRETS = {
    "anything",
    "placeholder",
    "password",
    "secret123",
    "password123",
    "your_password",
    "my_password",
    "admin123",
    "example123",
    "changeme",
    "change_me",
    "dummy_secret",
    "dummy_password",
    "supersecretkey",
    "supersecretkey12345",
}

# Patterns that may indicate secrets or sensitive data
# These are intentionally conservative to reduce false positives on code examples.
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{20,}", re.MULTILINE),
    # Match password/secret only when followed by an actual credential value (alnum string),
    # not code like `request.form['password']`
    re.compile(
        r"(?i)(secret|password|passwd|pwd)\s*[:=]\s*['\"]"
        r"[a-zA-Z0-9_\-!@#$%^&*]{8,}['\"]",
        re.MULTILINE,
    ),
    re.compile(r"(?i)(token|bearer)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{20,}", re.MULTILINE),
    re.compile(r"(?i)aws[_-]?(secret|access)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}"),
    re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub personal access token
    re.compile(r"sk-[a-zA-Z0-9]{32,}"),  # OpenAI-style API key
    re.compile(r"(?i)(client_secret|client_id)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}"),
]

# Patterns that may indicate PII
PII_PATTERNS = [
    # email (exclude framework decorators like @app., @router., @bp., @blueprint., @pytest., @api.)
    re.compile(r"\b[A-Za-z0-9._%+-]+@(?!app\.|router\.|bp\.|blueprint\.|pytest\.|api\.)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    # Credit card: require at least 13 consecutive digits (with optional spaces/dashes),
    # but not inside curly braces (which would be regex quantifiers)
    re.compile(r"(?<!\{)\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{1,4}\b(?!\})"),
]


@dataclass
class ValidationIssue:
    """A single validation issue found in an example."""

    line_number: int
    severity: str  # "error", "warning", "info"
    category: str
    message: str


@dataclass
class ValidationReport:
    """Complete validation report for a dataset file."""

    file_path: str
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    duplicate_count: int = 0
    potential_secrets: int = 0
    potential_pii: int = 0
    missing_provenance: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_critical_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def add_issue(
        self,
        line_number: int,
        severity: str,
        category: str,
        message: str,
    ) -> None:
        self.issues.append(ValidationIssue(line_number, severity, category, message))

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "",
            "=" * 60,
            "  DATASET VALIDATION REPORT",
            "=" * 60,
            f"  File:               {self.file_path}",
            f"  Examples:           {self.total_examples}",
            f"  Valid:              {self.valid_examples}",
            f"  Invalid:            {self.invalid_examples}",
            f"  Duplicates:         {self.duplicate_count}",
            f"  Potential secrets:  {self.potential_secrets}",
            f"  Potential PII:      {self.potential_pii}",
            f"  Missing provenance: {self.missing_provenance}",
            "=" * 60,
        ]

        if self.issues:
            lines.append("")
            lines.append("  ISSUES:")
            lines.append("-" * 60)
            for issue in self.issues[:50]:  # Cap display at 50
                lines.append(
                    f"  [{issue.severity.upper():7s}] Line {issue.line_number:5d}: "
                    f"[{issue.category}] {issue.message}"
                )
            if len(self.issues) > 50:
                lines.append(f"  ... and {len(self.issues) - 50} more issues")

        lines.append("")
        status = "FAIL" if self.has_critical_errors else "PASS"
        lines.append(f"  STATUS: {status}")
        lines.append("=" * 60)
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON output."""
        return {
            "file_path": self.file_path,
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "invalid_examples": self.invalid_examples,
            "duplicate_count": self.duplicate_count,
            "potential_secrets": self.potential_secrets,
            "potential_pii": self.potential_pii,
            "missing_provenance": self.missing_provenance,
            "has_critical_errors": self.has_critical_errors,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "line": i.line_number,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


def _check_secrets(content: str) -> list[str]:
    """Check content for potential secrets. Returns list of pattern descriptions."""
    found = []
    for pattern in SECRET_PATTERNS:
        matches = pattern.finditer(content)
        for match in matches:
            matched_str = match.group(0).lower()
            if any(dummy in matched_str for dummy in DUMMY_SECRETS):
                continue
            found.append(pattern.pattern[:50])
            break
    return found


def _check_pii(content: str) -> list[str]:
    """Check content for potential PII. Returns list of pattern descriptions."""
    found = []
    for pattern in PII_PATTERNS:
        if pattern.search(content):
            found.append(pattern.pattern[:50])
    return found


def _content_hash(example_dict: dict) -> str:
    """Compute a hash of the message contents for duplicate detection."""
    messages = example_dict.get("messages", [])
    content_parts = []
    for msg in messages:
        content_parts.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
    combined = "\n".join(content_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _check_invalid_unicode(text: str) -> bool:
    """Check for problematic Unicode characters."""
    # Check for null bytes, replacement characters, or private use area
    if "\x00" in text:
        return True
    if "\ufffd" in text:  # replacement character
        return True
    return False


def validate_file(
    file_path: str | Path,
    max_message_length: int = 16384,
    min_message_length: int = 10,
    max_sequence_length: int = 8192,
) -> ValidationReport:
    """Validate a JSONL dataset file.

    Args:
        file_path: Path to the JSONL file to validate.
        max_message_length: Maximum allowed length for a single message content.
        min_message_length: Minimum allowed length for user/assistant content.
        max_sequence_length: Maximum total character length for all messages combined.

    Returns:
        A ValidationReport with all findings.
    """
    file_path = Path(file_path)
    report = ValidationReport(file_path=str(file_path))

    if not file_path.exists():
        report.add_issue(0, "error", "file", f"File not found: {file_path}")
        return report

    if not file_path.suffix == ".jsonl":
        report.add_issue(0, "warning", "file", f"Expected .jsonl extension, got: {file_path.suffix}")

    seen_hashes: dict[str, int] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            report.total_examples += 1

            # 1. Valid JSON check
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                report.invalid_examples += 1
                report.add_issue(line_num, "error", "json", f"Invalid JSON: {e}")
                continue

            # 2. Schema validation via Pydantic
            try:
                example = TrainingExample.model_validate(data)
            except ValidationError as e:
                report.invalid_examples += 1
                errors = "; ".join(
                    f"{err['loc']}: {err['msg']}" for err in e.errors()[:3]
                )
                report.add_issue(line_num, "error", "schema", f"Schema error: {errors}")
                continue

            # If we get here, schema is valid
            is_valid = True

            # 3. Check message content lengths
            total_length = 0
            for msg in example.messages:
                content_len = len(msg.content)
                total_length += content_len

                if content_len > max_message_length:
                    report.add_issue(
                        line_num,
                        "warning",
                        "length",
                        f"Message content exceeds {max_message_length} chars "
                        f"(role={msg.role.value}, len={content_len})",
                    )

                if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    if content_len < min_message_length:
                        report.add_issue(
                            line_num,
                            "warning",
                            "length",
                            f"Very short message (role={msg.role.value}, len={content_len})",
                        )

            if total_length > max_sequence_length:
                report.add_issue(
                    line_num,
                    "warning",
                    "length",
                    f"Total message length ({total_length}) exceeds {max_sequence_length}",
                )

            # 4. Check for empty assistant responses
            for msg in example.messages:
                if msg.role == MessageRole.ASSISTANT and not msg.content.strip():
                    report.add_issue(
                        line_num, "error", "content", "Empty assistant response"
                    )
                    is_valid = False

            # 5. Check for secrets
            full_text = " ".join(m.content for m in example.messages)
            secrets = _check_secrets(full_text)
            if secrets:
                report.potential_secrets += 1
                report.add_issue(
                    line_num,
                    "error",
                    "secrets",
                    f"Potential secret detected (patterns: {len(secrets)})",
                )
                is_valid = False

            # 6. Check for PII
            pii = _check_pii(full_text)
            if pii:
                report.potential_pii += 1
                report.add_issue(
                    line_num,
                    "warning",
                    "pii",
                    f"Potential PII detected (patterns: {len(pii)})",
                )

            # 7. Check for invalid Unicode
            if _check_invalid_unicode(full_text):
                report.add_issue(
                    line_num,
                    "warning",
                    "unicode",
                    "Invalid or problematic Unicode characters detected",
                )

            # 8. Check provenance
            meta = example.metadata
            if not meta.source and not meta.synthetic:
                report.missing_provenance += 1
                report.add_issue(
                    line_num,
                    "warning",
                    "provenance",
                    "Non-synthetic example missing 'source' in metadata",
                )

            # 9. Check training permission
            if meta.allowed_for_training is None and not meta.synthetic:
                report.add_issue(
                    line_num,
                    "info",
                    "license",
                    "Training permission unknown (allowed_for_training=None)",
                )

            # 10. Duplicate detection
            content_hash = _content_hash(data)
            if content_hash in seen_hashes:
                report.duplicate_count += 1
                report.add_issue(
                    line_num,
                    "warning",
                    "duplicate",
                    f"Duplicate of line {seen_hashes[content_hash]}",
                )
            else:
                seen_hashes[content_hash] = line_num

            if is_valid:
                report.valid_examples += 1
            else:
                report.invalid_examples += 1

    return report


def validate_file_cli(args: Optional[list[str]] = None) -> int:
    """CLI entry point for dataset validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate a CyberCodeMini JSONL dataset")
    parser.add_argument("file", type=str, help="Path to the JSONL file to validate")
    parser.add_argument("--max-msg-len", type=int, default=16384, help="Max message length")
    parser.add_argument("--min-msg-len", type=int, default=10, help="Min message length")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Max total sequence length")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parsed = parser.parse_args(args)

    report = validate_file(
        parsed.file,
        max_message_length=parsed.max_msg_len,
        min_message_length=parsed.min_msg_len,
        max_sequence_length=parsed.max_seq_len,
    )

    if parsed.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    return 1 if report.has_critical_errors else 0
