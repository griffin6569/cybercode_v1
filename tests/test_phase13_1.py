"""CyberCodeMini Phase 13.1 Quota Request Audit Test Suite

Verifies Phase 13.1 quota request artifacts, status determination, zero-compute controls,
and quota calculations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

QUOTA_JSON_PATH = DATA_DIR / "metadata" / "azure_phase13_1_quota_request.json"
QUOTA_MD_PATH = DATA_DIR / "metadata" / "azure_phase13_1_quota_request.md"


def test_1_quota_request_artifacts_exist():
    """Test 1: Quota request JSON and Markdown artifacts exist."""
    assert QUOTA_JSON_PATH.exists(), "azure_phase13_1_quota_request.json does not exist"
    assert QUOTA_MD_PATH.exists(), "azure_phase13_1_quota_request.md does not exist"


def test_2_quota_request_status():
    """Test 2: Status is MANUAL_QUOTA_REQUEST_REQUIRED."""
    data = json.loads(QUOTA_JSON_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "MANUAL_QUOTA_REQUEST_REQUIRED"
    assert data["previous_quota_vcpus"] == 0
    assert data["requested_quota_vcpus"] == 36
    assert data["quota_family"] == "StandardNVADSA10v5Family"


def test_3_zero_compute_and_credit_safety():
    """Test 3: Zero GPU compute, zero training jobs, $0 credit spent."""
    data = json.loads(QUOTA_JSON_PATH.read_text(encoding="utf-8"))
    assert data["gpu_compute_count"] == 0
    assert data["training_job_count"] == 0
    assert data["azure_credit_spent_usd"] == 0.0


def test_4_frozen_dataset_integrity():
    """Test 4: Frozen dataset SHA-256 matches expected v0.3.0 hash."""
    data = json.loads(QUOTA_JSON_PATH.read_text(encoding="utf-8"))
    assert data["frozen_dataset_version"] == "v0.3.0"
    assert data["frozen_dataset_sha256"] == "c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5"
