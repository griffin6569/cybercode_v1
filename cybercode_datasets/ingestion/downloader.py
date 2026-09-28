"""CyberCodeMini Safe Dataset Downloader & Data Fetcher

Fetches dataset splits securely as untrusted raw data.
Strictly prevents code execution of downloaded scripts/repositories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence


class SafeDatasetDownloader:
    """Downloader operating on data only, preventing arbitrary code execution."""

    def fetch_local_or_cached_jsonl(self, file_path: Path | str, max_rows: int = 500) -> list[dict[str, Any]]:
        """Safely load rows from a local JSONL/JSON data file."""
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        rows = []
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
                        if len(rows) >= max_rows:
                            break
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows = data[:max_rows]
                elif isinstance(data, dict):
                    rows = [data]

        return rows
