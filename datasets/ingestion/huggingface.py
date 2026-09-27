"""CyberCodeMini Hugging Face Ingestion Engine (Phase 10)

Inspects, samples, verifies licensing/provenance, and ingests Hugging Face datasets deterministically.
Pinnable revisions, bounded sampling (max 500 rows), and inspection report generation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from datasets.adapters.hf_conversational_adapter import HuggingFaceConversationalAdapter
from datasets.adapters.security_review_adapter import SecurityReviewAdapter
from datasets.adapters.trajectory_adapter import AgentTrajectoryAdapter
from datasets.ingestion.license import LicensePolicyValidator
from datasets.ingestion.metadata import ContentClassifier
from datasets.ingestion.provenance import create_provenance_record
from datasets.ingestion.registry import IngestionRegistry
from datasets.schemas.schema import TrainingExample


class HuggingFaceIngestionEngine:
    """Ingestion engine for Hugging Face datasets with pinned revisions and bounded sampling."""

    def __init__(self, registry: Optional[IngestionRegistry] = None) -> None:
        self.registry = registry or IngestionRegistry()
        self.license_validator = LicensePolicyValidator()
        self.classifier = ContentClassifier()

    def _sample_rows(self, raw_rows: list[dict[str, Any]], max_rows: int = 500, seed: int = 42) -> list[dict[str, Any]]:
        """Sample rows deterministically using a fixed seed."""
        if len(raw_rows) <= max_rows:
            return raw_rows
        
        rng = random.Random(seed)
        shuffled = list(raw_rows)
        rng.shuffle(shuffled)
        return shuffled[:max_rows]

    def inspect_dataset(
        self,
        dataset_id: str,
        revision: str = "main",
        split: str = "train",
        subset: Optional[str] = None,
        max_inspect_rows: int = 500,
        license_name: str = "unknown",
        license_source: str = "dataset_card",
    ) -> dict[str, Any]:
        """Inspect a dataset candidate and generate data/metadata/ingestion_reports/<dataset>.json."""
        lic_result = self.license_validator.verify_license(license_name, license_source=license_source)

        report = {
            "dataset_id": dataset_id,
            "revision": revision,
            "split": split,
            "subset": subset,
            "rows_inspected": max_inspect_rows,
            "license": license_name,
            "license_verified": lic_result.verified,
            "commercial_restriction": not lic_result.is_commercial,
            "license_status": lic_result.status,
            "approval_status": lic_result.status,
            "training_candidate": lic_result.status == "approved",
            "evaluation_candidate": True,
            "notes": lic_result.source_notes,
        }

        # Save report
        safe_name = dataset_id.replace("/", "_")
        report_path = Path("data/metadata/ingestion_reports") / f"{safe_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def ingest_raw_rows(
        self,
        dataset_id: str,
        raw_rows: list[dict[str, Any]],
        revision: str = "main",
        split: str = "train",
        license_name: str = "unknown",
        license_source: str = "dataset_card",
        max_rows: int = 500,
        seed: int = 42,
    ) -> tuple[list[TrainingExample], dict[str, Any]]:
        """Ingest, verify, sample, and convert raw dataset rows."""
        # 1. Inspect license policy
        inspection = self.inspect_dataset(
            dataset_id=dataset_id,
            revision=revision,
            split=split,
            max_inspect_rows=len(raw_rows),
            license_name=license_name,
            license_source=license_source,
        )

        # 2. Register candidate in registry
        entry = self.registry.register_or_update(
            dataset_id=dataset_id,
            revision=revision,
            license_name=license_name,
            license_source=license_source,
            provenance_status="verified" if inspection["license_verified"] else "unclear",
            security_content="defensive",
            authorization_status="defensive" if inspection["license_verified"] else "unknown",
            sample_count=len(raw_rows),
        )

        # 3. Sample rows deterministically
        sampled = self._sample_rows(raw_rows, max_rows=max_rows, seed=seed)

        # 4. Select appropriate adapter
        if "swe-bench" in dataset_id.lower():
            adapter = AgentTrajectoryAdapter(source_name=dataset_id, registry=self.registry)
        elif "secbench" in dataset_id.lower():
            adapter = SecurityReviewAdapter(source_name=dataset_id, registry=self.registry)
        else:
            adapter = HuggingFaceConversationalAdapter(source_name=dataset_id, registry=self.registry)

        converted_examples = adapter.convert_batch(sampled)

        # Attach provenance to converted examples
        for ex in converted_examples:
            ex.metadata.source = dataset_id
            ex.metadata.license = license_name
            ex.metadata.allowed_for_training = entry.approval_status == "approved"

        return converted_examples, inspection
