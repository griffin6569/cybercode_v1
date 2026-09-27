"""CyberCodeMini Phase 13.3 Support Request Draft Test Suite

Verifies Phase 13.3 support request artifacts, draft status determination,
quota parameters, zero-compute controls, and factual representation guidelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SUPPORT_JSON_PATH = DATA_DIR / "metadata" / "azure_phase13_3_support_request.json"
SUPPORT_MD_PATH = DATA_DIR / "metadata" / "azure_phase13_3_support_request.md"


def test_1_support_request_artifacts_exist():
    """Test 1: Phase 13.3 JSON and Markdown artifacts exist."""
    assert SUPPORT_JSON_PATH.exists(), "azure_phase13_3_support_request.json does not exist"
    assert SUPPORT_MD_PATH.exists(), "azure_phase13_3_support_request.md does not exist"


def test_2_support_request_status_and_parameters():
    """Test 2: Status is SUPPORT_REQUEST_DRAFT_READY and quota request is 36 vCPUs."""
    data = json.loads(SUPPORT_JSON_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "SUPPORT_REQUEST_DRAFT_READY"
    assert data["subscription"]["subscription_id"] == "4d6ad347-4459-4206-ac7c-54cc59637a4a"
    
    details = data["support_request_details"]
    assert details["problem_type"] == "Service and subscription limits (quotas)"
    assert details["service"] == "Azure Machine Learning"
    assert details["region"] == "South Africa North"
    assert details["quota_family"] == "StandardNVADSA10v5Family"
    assert details["requested_quota_vcpus"] == 36
    assert details["current_quota_vcpus"] == 0
    assert details["target_gpu_sku"] == "Standard_NV36ads_A10_v5"


def test_3_factual_body_text_content():
    """Test 3: Factual body contains required accurate statements."""
    data = json.loads(SUPPORT_JSON_PATH.read_text(encoding="utf-8"))
    body = data["support_request_details"]["factual_request_body"]
    assert "Azure for Students" in body
    assert "CyberCodeMini" in body
    assert "Standard_NV36ads_A10_v5" in body
    assert "36 vCPUs" in body
    assert "Owner RBAC" in body
    assert "ResourceNotAvailableForOffer" in body


def test_4_zero_compute_and_credit_safety():
    """Test 4: Zero GPU compute, zero training jobs, zero VMs, $0 credit spent, subscription unchanged."""
    data = json.loads(SUPPORT_JSON_PATH.read_text(encoding="utf-8"))
    cost_status = data["cost_protection_status"]
    assert cost_status["subscription_changed"] is False
    assert cost_status["gpu_compute_created"] == 0
    assert cost_status["vms_created"] == 0
    assert cost_status["aml_jobs_submitted"] == 0
    assert cost_status["azure_credit_spent_usd"] == 0.0


def test_5_frozen_dataset_integrity():
    """Test 5: Frozen dataset SHA-256 matches expected v0.3.0 hash."""
    data = json.loads(SUPPORT_JSON_PATH.read_text(encoding="utf-8"))
    frozen = data["frozen_dataset"]
    assert frozen["version"] == "v0.3.0"
    assert frozen["sha256"] == "c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5"
