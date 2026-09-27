"""CyberCodeMini Phase 13.2 Student Subscription GPU Audit Test Suite

Verifies Phase 13.2 student subscription GPU eligibility audit artifacts, RBAC role verification,
status determination, subscription offer policy blocker analysis, and zero-compute safety controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BLOCKER_JSON_PATH = DATA_DIR / "metadata" / "azure_phase13_2_students_gpu_blocker.json"
BLOCKER_MD_PATH = DATA_DIR / "metadata" / "azure_phase13_2_students_gpu_blocker.md"


def test_1_blocker_artifacts_exist():
    """Test 1: Phase 13.2 JSON and Markdown artifacts exist."""
    assert BLOCKER_JSON_PATH.exists(), "azure_phase13_2_students_gpu_blocker.json does not exist"
    assert BLOCKER_MD_PATH.exists(), "azure_phase13_2_students_gpu_blocker.md does not exist"


def test_2_blocker_status_and_rbac():
    """Test 2: Status is STUDENT_SUBSCRIPTION_GPU_BLOCKED and user is Owner."""
    data = json.loads(BLOCKER_JSON_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "STUDENT_SUBSCRIPTION_GPU_BLOCKED"
    assert data["user_rbac_role"] == "Owner"
    assert data["current_quota_vcpus"] == 0
    assert data["required_quota_vcpus"] == 36
    assert data["quota_family"] == "StandardNVADSA10v5Family"
    assert data["subscription"]["subscription_id"] == "4d6ad347-4459-4206-ac7c-54cc59637a4a"


def test_3_subscription_policy_not_rbac_blocker():
    """Test 3: Quota API result is ResourceNotAvailableForOffer."""
    data = json.loads(BLOCKER_JSON_PATH.read_text(encoding="utf-8"))
    assert data["quota_api_result"] == "ResourceNotAvailableForOffer"
    assert "Azure for Students" in data["subscription"]["name"]


def test_4_zero_compute_and_credit_safety():
    """Test 4: Zero GPU compute, zero training jobs, zero VMs, $0 credit spent, subscription unchanged."""
    data = json.loads(BLOCKER_JSON_PATH.read_text(encoding="utf-8"))
    cost_status = data["cost_protection_status"]
    assert cost_status["subscription_changed"] is False
    assert cost_status["gpu_compute_created"] == 0
    assert cost_status["vms_created"] == 0
    assert cost_status["aml_jobs_submitted"] == 0
    assert cost_status["azure_credit_spent_usd"] == 0.0


def test_5_frozen_dataset_integrity():
    """Test 5: Frozen dataset SHA-256 matches expected v0.3.0 hash."""
    data = json.loads(BLOCKER_JSON_PATH.read_text(encoding="utf-8"))
    frozen = data["frozen_dataset"]
    assert frozen["version"] == "v0.3.0"
    assert frozen["sha256"] == "c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5"
