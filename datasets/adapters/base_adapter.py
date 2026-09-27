"""CyberCodeMini Base Dataset Adapter Interface

Conceptual interface:
DatasetAdapter
├── inspect()
├── convert()
├── validate()
└── provenance()

All specific adapters inherit from DatasetAdapter and implement format conversion,
provenance metadata tagging, source ID preservation, and validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from datasets.provenance.provenance import DatasetRegistry, DatasetRegistryEntry
from datasets.schemas.schema import TrainingExample
from datasets.validators.validator import ValidationReport, validate_file


class DatasetAdapter(ABC):
    """Abstract base adapter interface matching CyberCodeMini architecture."""

    def __init__(
        self,
        source_name: str,
        registry: Optional[DatasetRegistry] = None,
        default_synthetic: bool = False,
    ) -> None:
        self.source_name = source_name
        self.registry = registry or DatasetRegistry()
        self.default_synthetic = default_synthetic
        
        # Look up registry entry
        self.registry_entry: Optional[DatasetRegistryEntry] = self.registry.get_entry(source_name)
        if self.registry_entry:
            self.allowed_for_training = self.registry_entry.training_allowed
            self.license = self.registry_entry.license
        else:
            # Excluded by default if not in registry
            self.allowed_for_training = False
            self.license = "unknown"

    def inspect(self, raw_entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Inspect a sample batch of raw entries and return dataset stats."""
        total = len(raw_entries)
        sample_keys = list(raw_entries[0].keys()) if total > 0 else []
        return {
            "source_name": self.source_name,
            "sample_count": total,
            "sample_keys": sample_keys,
            "allowed_for_training": self.allowed_for_training,
            "license": self.license,
        }

    @abstractmethod
    def convert(self, raw_entry: dict[str, Any]) -> TrainingExample:
        """Convert a single raw entry into a canonical TrainingExample."""
        pass

    def convert_batch(self, raw_entries: Sequence[dict[str, Any]]) -> list[TrainingExample]:
        """Convert a batch of raw entries into canonical TrainingExample instances."""
        converted = []
        for idx, entry in enumerate(raw_entries):
            try:
                example = self.convert(entry)
                converted.append(example)
            except Exception as err:
                print(f"[{self.__class__.__name__}] Item #{idx} conversion skipped: {err}")
        return converted

    def validate(self, example_or_file: TrainingExample | str) -> bool:
        """Validate an converted example or JSONL file path."""
        if isinstance(example_or_file, str):
            report: ValidationReport = validate_file(example_or_file)
            return not report.has_critical_errors
        else:
            # Validate individual TrainingExample object
            try:
                TrainingExample.model_validate(example_or_file.model_dump())
                return True
            except Exception:
                return False

    def provenance(self) -> dict[str, Any]:
        """Return dataset provenance metadata."""
        if self.registry_entry:
            return self.registry_entry.to_dict()
        return {
            "source_name": self.source_name,
            "license": self.license,
            "allowed_for_training": self.allowed_for_training,
            "status": "review" if not self.allowed_for_training else "approved",
        }

    def _extract_source_id(self, raw_entry: dict[str, Any], default_prefix: str = "item") -> str:
        """Extract or generate a stable source ID from raw entry."""
        for key in ("id", "source_id", "cve_id", "task_id", "sample_id", "uuid", "hash"):
            if key in raw_entry and raw_entry[key]:
                return str(raw_entry[key])
        raw_repr = str(sorted(raw_entry.items()))
        short_hash = DatasetRegistry.compute_sha256(raw_repr)[:12]
        return f"{default_prefix}_{short_hash}"


# Alias for backward compatibility
BaseAdapter = DatasetAdapter
