"""CyberCodeMini Ingestion Provenance Manager

Manages example-level provenance records to ensure source attribution, dataset ID,
revision, split, and license metadata survive pipeline transformations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ExampleProvenance:
    """Provenance record attached to each ingested example."""

    source_dataset: str
    source_id: str
    source_revision: str = "main"
    source_split: str = "train"
    license: str = "unknown"
    license_verified: bool = False
    provenance_notes: str = ""
    original_metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_provenance_record(
    source_dataset: str,
    source_id: str,
    revision: str = "main",
    split: str = "train",
    license_name: str = "unknown",
    license_verified: bool = False,
    notes: str = "",
    original_metadata: Optional[dict[str, Any]] = None,
) -> ExampleProvenance:
    """Create a standardized ExampleProvenance record."""
    return ExampleProvenance(
        source_dataset=source_dataset,
        source_id=source_id,
        source_revision=revision,
        source_split=split,
        license=license_name,
        license_verified=license_verified,
        provenance_notes=notes,
        original_metadata=original_metadata or {},
    )
