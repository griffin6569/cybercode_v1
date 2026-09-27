"""CyberCodeMini Expanded Dataset Registry Manager

Manages dataset registration, license verification tracking, provenance state,
and approval workflows in data/metadata/dataset_registry.json.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from datasets.ingestion.license import LicensePolicyValidator


@dataclass
class ExpandedRegistryEntry:
    """Expanded dataset registry schema matching CyberCodeMini Phase 10 requirements."""

    dataset_id: str
    revision: str = "main"
    source_url: str = ""
    license: str = "unknown"
    license_verified: bool = False
    license_source: str = "unverified"
    usage_restrictions: list[str] = field(default_factory=list)
    commercial_use: str = "unknown"  # "allowed" | "restricted" | "unknown"
    provenance_status: str = "unknown"  # "verified" | "unclear" | "unknown"
    security_content: str = "unknown"  # "defensive" | "benchmark" | "dual_use" | "offensive" | "unknown"
    authorization_status: str = "unknown"  # "authorized" | "explicit" | "defensive" | "unknown"
    quality_status: str = "unreviewed"  # "passed" | "low_quality" | "unreviewed"
    approval_status: str = "review"  # "approved" | "review" | "rejected"
    added_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    )
    review_notes: str = ""
    sample_count: int = 0

    @property
    def training_allowed(self) -> bool:
        return self.approval_status == "approved" and self.license_verified

    def __post_init__(self) -> None:
        """Enforce strict default rules:
        - If license is unknown or unverified -> approval_status = "review"
        - If provenance is unclear -> approval_status = "review"
        - Do not allow automatic promotion of unreviewed datasets to approved
        """
        if not self.license_verified or self.license.lower() in ("unknown", "unclear", "none"):
            if self.approval_status == "approved":
                self.approval_status = "review"
                self.review_notes += " Reverted to review: License is unverified or unknown."

        if self.provenance_status in ("unclear", "unknown") and self.approval_status == "approved":
            self.approval_status = "review"
            self.review_notes += " Reverted to review: Provenance is unclear."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionRegistry:
    """Registry manager stored in data/metadata/dataset_registry.json."""

    def __init__(self, registry_file: Optional[Path | str] = None) -> None:
        if registry_file is None:
            registry_file = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "metadata"
                / "dataset_registry.json"
            )
        self.registry_file = Path(registry_file)
        self.entries: dict[str, ExpandedRegistryEntry] = {}
        self.license_validator = LicensePolicyValidator()
        self.load()

    def load(self) -> None:
        """Load registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, entry_dict in data.items():
                        # Support legacy schema dicts gracefully
                        mapped = {
                            "dataset_id": entry_dict.get("dataset_id") or entry_dict.get("name") or key,
                            "revision": entry_dict.get("revision", "main"),
                            "source_url": entry_dict.get("source_url", ""),
                            "license": entry_dict.get("license", "unknown"),
                            "license_verified": entry_dict.get("license_verified", False),
                            "license_source": entry_dict.get("license_source", "unverified"),
                            "usage_restrictions": entry_dict.get("usage_restrictions", []),
                            "commercial_use": entry_dict.get("commercial_use", "unknown"),
                            "provenance_status": entry_dict.get("provenance_status", "unknown"),
                            "security_content": entry_dict.get("security_content", "unknown"),
                            "authorization_status": entry_dict.get("authorization_status", "unknown"),
                            "quality_status": entry_dict.get("quality_status", "unreviewed"),
                            "approval_status": entry_dict.get("approval_status") or entry_dict.get("status", "review"),
                            "added_at": entry_dict.get("added_at") or entry_dict.get("date_reviewed", "2026-09-27"),
                            "review_notes": entry_dict.get("review_notes") or entry_dict.get("provenance_notes", ""),
                            "sample_count": entry_dict.get("sample_count", 0),
                        }
                        self.entries[mapped["dataset_id"]] = ExpandedRegistryEntry(**mapped)
            except Exception as e:
                print(f"Warning: Failed to load expanded registry: {e}")

    def save(self) -> None:
        """Save registry to JSON file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self.entries.items()}, f, indent=2)

    def register_or_update(
        self,
        dataset_id: str,
        revision: str = "main",
        source_url: str = "",
        license_name: str = "unknown",
        license_source: str = "unverified",
        provenance_status: str = "unknown",
        security_content: str = "unknown",
        authorization_status: str = "unknown",
        review_notes: str = "",
        sample_count: int = 0,
    ) -> ExpandedRegistryEntry:
        """Register or update a dataset entry with license policy evaluation."""
        lic_result = self.license_validator.verify_license(license_name, license_source=license_source)
        
        entry = ExpandedRegistryEntry(
            dataset_id=dataset_id,
            revision=revision,
            source_url=source_url,
            license=license_name,
            license_verified=lic_result.verified,
            license_source=license_source,
            commercial_use="allowed" if lic_result.is_commercial else "restricted",
            provenance_status=provenance_status,
            security_content=security_content,
            authorization_status=authorization_status,
            approval_status=lic_result.status,
            review_notes=review_notes or lic_result.source_notes,
            sample_count=sample_count,
        )
        self.entries[dataset_id] = entry
        self.save()
        return entry

    def is_approved_for_training(self, dataset_id: str) -> bool:
        """Check if dataset is explicitly APPROVED for model training."""
        entry = self.entries.get(dataset_id)
        if entry is None:
            return False
        return entry.approval_status == "approved" and entry.license_verified

    def get_entry(self, dataset_id: str) -> Optional[ExpandedRegistryEntry]:
        return self.entries.get(dataset_id)
