"""CyberCodeMini Phase 13 Preflight Automated Test Suite

Verifies Azure Phase 13 preflight audit data, zero-compute credit safety constraints,
quota calculations, and report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PREFLIGHT_JSON_PATH = DATA_DIR / "metadata" / "azure_phase13_preflight.json"
PREFLIGHT_MD_PATH = DATA_DIR / "metadata" / "azure_phase13_preflight.md"


def test_1_preflight_artifacts_exist():
    """Test 1: Preflight JSON and Markdown reports exist."""
    assert PREFLIGHT_JSON_PATH.exists(), "azure_phase13_preflight.json does not exist"
    assert PREFLIGHT_MD_PATH.exists(), "azure_phase13_preflight.md does not exist"


def test_2_subscription_and_workspace_metadata():
    """Test 2: Preflight JSON contains accurate subscription and workspace metadata."""
    data = json.loads(PREFLIGHT_JSON_PATH.read_text(encoding="utf-8"))
    assert data["subscription"]["subscription_id"] == "4d6ad347-4459-4206-ac7c-54cc59637a4a"
    assert data["workspace"]["workspace_name"] == "cybercodemini-ml"
    assert data["workspace"]["workspace_region"] == "southafricanorth"
    assert data["workspace"]["resource_group"] == "cybercodemini-rg"


def test_3_provider_registrations():
    """Test 3: Provider registrations are all Registered."""
    data = json.loads(PREFLIGHT_JSON_PATH.read_text(encoding="utf-8"))
    providers = data["providers"]
    assert providers["Microsoft.Compute"] == "Registered"
    assert providers["Microsoft.MachineLearningServices"] == "Registered"
    assert providers["Microsoft.Quota"] == "Registered"


def test_4_quota_calculation_and_status():
    """Test 4: Quota calculation is exactly 36 vCPUs and status is QUOTA_REQUEST_REQUIRED."""
    data = json.loads(PREFLIGHT_JSON_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "QUOTA_REQUEST_REQUIRED"
    assert data["current_quota"]["limit_value"] == 0
    assert data["minimum_quota_required"]["target_vcpus"] == 36


def test_5_credit_safety_gate():
    """Test 5: Zero GPU compute, zero training jobs, $0 credit spent."""
    data = json.loads(PREFLIGHT_JSON_PATH.read_text(encoding="utf-8"))
    safety = data["safety_verifications"]
    assert safety["azure_resources_created"] == 0
    assert safety["azure_training_jobs_submitted"] == 0
    assert safety["azure_credit_spent_usd"] == 0.0


def test_6_estimated_pilot_cost():
    """Test 6: Verified hourly rate ($1.85) and estimated pilot cost (< $1.00)."""
    data = json.loads(PREFLIGHT_JSON_PATH.read_text(encoding="utf-8"))
    cost = data["cost_and_pricing"]
    assert cost["hourly_price_usd"] == 1.85
    assert cost["estimated_pilot_cost_usd"] < 1.00
