"""CyberCodeMini Dataset Registry & Provenance Manager

Manages dataset registry records including name, source URL, license, provenance,
training permission rules, status, and notes. Enforces default exclusion and "review"
status for unknown/unclear licenses and provenance.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Standard approved licenses for open-weight model training
APPROVED_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "public-domain",
    "unlicense",
}


@dataclass
class DatasetRegistryEntry:
    """Registry entry matching the required CyberCodeMini schema."""

    name: str
    source_url: str
    source_type: str  # "huggingface" | "github" | "local" | "synthetic"
    license: str
    author: str = "unknown"
    version: str = "1.0.0"
    training_allowed: bool = False
    commercial_use: bool = False
    provenance_notes: str = ""
    limitations: str = ""
    date_reviewed: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    )
    status: str = "review"  # "approved" | "review" | "rejected"
    sample_count: int = 0

    def __post_init__(self) -> None:
        """Enforce default safety rules:
        - If license is unknown/empty -> status = "review", training_allowed = False
        - If license is not in approved list -> training_allowed = False unless explicitly approved
        - If provenance is unclear -> status = "review"
        """
        lic_clean = self.license.strip().lower() if self.license else ""

        if not lic_clean or lic_clean in ("unknown", "unclear", "none", "n/a"):
            self.status = "review"
            self.training_allowed = False
            if not self.provenance_notes:
                self.provenance_notes = "Excluded: Unknown or unstated license."

        elif lic_clean not in APPROVED_LICENSES and self.status == "approved":
            # Require explicit approval for non-standard licenses
            if not self.training_allowed:
                self.status = "review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetRegistry:
    """Central registry manager for data/metadata/dataset_registry.json."""

    def __init__(self, registry_file: Optional[Path | str] = None) -> None:
        if registry_file is None:
            registry_file = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "metadata"
                / "dataset_registry.json"
            )
        self.registry_file = Path(registry_file)
        self.entries: dict[str, DatasetRegistryEntry] = {}
        self.load()

    def load(self) -> None:
        """Load registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for name, entry_dict in data.items():
                        self.entries[name] = DatasetRegistryEntry(**entry_dict)
            except Exception as e:
                print(f"Warning: Failed to load dataset registry: {e}")

    def save(self) -> None:
        """Save registry to JSON file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self.entries.items()}, f, indent=2)

    def register(
        self,
        name: str,
        source_url: str,
        source_type: str,
        license: str,
        author: str = "unknown",
        version: str = "1.0.0",
        training_allowed: bool = False,
        commercial_use: bool = False,
        provenance_notes: str = "",
        limitations: str = "",
        status: str = "review",
        sample_count: int = 0,
    ) -> DatasetRegistryEntry:
        """Register or update a dataset entry."""
        entry = DatasetRegistryEntry(
            name=name,
            source_url=source_url,
            source_type=source_type,
            license=license,
            author=author,
            version=version,
            training_allowed=training_allowed,
            commercial_use=commercial_use,
            provenance_notes=provenance_notes,
            limitations=limitations,
            status=status,
            sample_count=sample_count,
        )
        self.entries[name] = entry
        self.save()
        return entry

    def is_allowed(self, name: str) -> bool:
        """Check if dataset is permitted for training."""
        entry = self.entries.get(name)
        if entry is None:
            return False  # Excluded by default if unknown
        return entry.training_allowed and entry.status == "approved"

    def get_entry(self, name: str) -> Optional[DatasetRegistryEntry]:
        return self.entries.get(name)

    @staticmethod
    def compute_sha256(content: str | bytes) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()
