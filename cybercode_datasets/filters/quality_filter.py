"""CyberCodeMini Quality Filter & Leakage Detector

Filters datasets by:
1. Deduplication (exact content & normalized content hashing)
2. Malformed conversation detection
3. Low-information / empty response filtering
4. Test set contamination detection (HumanEval, MBPP, SWE-bench test split, SecBench)
5. Routing flagged items to data/metadata/review_queue.jsonl
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from cybercode_datasets.schemas.schema import MessageRole, TrainingExample
from cybercode_datasets.validators.quality import score_example

# Known test benchmark prompt signatures to prevent benchmark data leakage / contamination
CONTAMINATION_SIGNATURES = [
    r"def has_close_elements",  # HumanEval
    r"def prompt_eval",
    r"def find_zero",
    r"HumanEval/",
    r"MBPP/",
    r"SWE-bench/test",
    r"SecBench/eval",
    r"benchmark_test_case",
]


class QualityFilter:
    """Dataset quality filter and contamination detector."""

    def __init__(
        self,
        min_quality_score: float = 70.0,
        review_queue_path: Optional[Path | str] = None,
    ) -> None:
        self.min_quality_score = min_quality_score
        if review_queue_path is None:
            review_queue_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "metadata"
                / "review_queue.jsonl"
            )
        self.review_queue_path = Path(review_queue_path)
        self.seen_hashes: set[str] = set()

    def is_contaminated(self, example: TrainingExample) -> tuple[bool, str]:
        """Check if an example contains benchmark test set contamination."""
        full_text = " ".join(m.content for m in example.messages)
        for sig in CONTAMINATION_SIGNATURES:
            if re.search(sig, full_text, re.IGNORECASE):
                return True, f"Matched benchmark contamination signature: '{sig}'"
        return False, ""

    def _compute_content_hash(self, example: TrainingExample) -> str:
        """Compute SHA256 hash of message role + content sequence."""
        parts = [f"{m.role.value}:{m.content.strip()}" for m in example.messages]
        combined = "\n".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def filter_example(self, example: TrainingExample) -> tuple[bool, str, list[str]]:
        """Filter an example.
        
        Returns:
            (keep: bool, reason: str, review_reasons: list[str])
        """
        # 1. Deduplication
        content_hash = self._compute_content_hash(example)
        if content_hash in self.seen_hashes:
            return False, "Duplicate example content", ["Duplicate content"]
        self.seen_hashes.add(content_hash)

        # 2. Contamination check
        is_contam, contam_reason = self.is_contaminated(example)
        if is_contam:
            return False, contam_reason, [contam_reason]

        # 3. Quality score check
        assessment = score_example(example)

        if assessment.requires_human_review:
            self._route_to_review_queue(example, assessment.review_reasons or assessment.reasons)

        if assessment.score < self.min_quality_score:
            return False, f"Low quality score ({assessment.score:.1f} < {self.min_quality_score})", assessment.reasons

        return True, "Passed", []

    def _route_to_review_queue(self, example: TrainingExample, reasons: list[str]) -> None:
        """Append flagged example to data/metadata/review_queue.jsonl."""
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "source_id": example.metadata.source_id or "unknown",
            "source": example.metadata.source,
            "category": example.metadata.category.value,
            "review_reasons": reasons,
            "example": example.model_dump(),
        }
        with open(self.review_queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def filter_batch(self, examples: Sequence[TrainingExample]) -> tuple[list[TrainingExample], list[dict[str, Any]]]:
        """Filter a batch of TrainingExamples.
        
        Returns:
            (approved_examples, rejected_records)
        """
        approved = []
        rejected = []

        for ex in examples:
            keep, reason, review_reasons = self.filter_example(ex)
            if keep:
                approved.append(ex)
            else:
                rejected.append({
                    "source_id": ex.metadata.source_id,
                    "reason": reason,
                    "review_reasons": review_reasons,
                })

        return approved, rejected
