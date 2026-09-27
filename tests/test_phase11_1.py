"""CyberCodeMini Phase 11.1 Forensic Audit Test Suite

Verifies exact row counts (900 train, 90 val, 200 eval, 50 review), capability taxonomy consistency,
agent trajectory structural classification, source totals, license totals, security authorization metadata,
tier separation, row-level schema validity, quality scores, capability gap script configuration,
and benchmark isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets.schemas.capabilities import Capability
from datasets.schemas.schema import TrainingExample
from scripts.analyze_capability_gaps import analyze_gaps

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_V030_PATH = DATA_DIR / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.3.0.jsonl"
VAL_V030_PATH = DATA_DIR / "processed" / "validation_candidates" / "cybercodemini_validation_candidates_v0.3.0.jsonl"
SECBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "secbench" / "examples.jsonl"
SWEBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "swe_bench" / "examples.jsonl"
REVIEW_PATH = DATA_DIR / "review" / "unknown_license" / "examples.jsonl"
AUDIT_JSON_PATH = DATA_DIR / "metadata" / "phase11_1_audit.json"
AGENT_AUDIT_JSON_PATH = DATA_DIR / "metadata" / "agent_trajectory_audit.json"


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"Path '{path}' does not exist"
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def test_1_current_corpus_count():
    """Test 1: Training corpus contains exactly 900 rows."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    assert len(train_items) == 900, f"Expected 900 training rows, got {len(train_items)}"


def test_2_validation_count():
    """Test 2: Validation corpus contains exactly 90 rows."""
    val_items = load_jsonl(VAL_V030_PATH)
    assert len(val_items) == 90, f"Expected 90 validation rows, got {len(val_items)}"


def test_3_evaluation_and_review_counts():
    """Test 3: SecBench (100), SWE-bench (100), and Review (50) counts match."""
    secbench_items = load_jsonl(SECBENCH_EVAL_PATH)
    swebench_items = load_jsonl(SWEBENCH_EVAL_PATH)
    review_items = load_jsonl(REVIEW_PATH)

    assert len(secbench_items) == 100
    assert len(swebench_items) == 100
    assert len(review_items) == 50


def test_4_capability_taxonomy_consistency():
    """Test 4: Every training item primary capability exists in canonical taxonomy."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    valid_caps = {c.value for c in Capability}
    for item in train_items:
        meta = item.get("metadata", {})
        cap = meta.get("primary_capability") or meta.get("category")
        assert cap in valid_caps, f"Invalid capability '{cap}' in item: {item}"


def test_5_agent_trajectory_classification():
    """Test 5: Agent trajectory audit report exists and records 147 full trajectories in training."""
    assert AGENT_AUDIT_JSON_PATH.exists()
    audit_data = json.loads(AGENT_AUDIT_JSON_PATH.read_text(encoding="utf-8"))
    assert audit_data["training_agent_examples"] == 147
    assert audit_data["training_breakdown"]["full_agent_trajectory"] == 147


def test_6_source_totals():
    """Test 6: Source distribution sums to 900 across all 8 sources."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    src_counts: dict[str, int] = {}
    for item in train_items:
        src = item["metadata"].get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    assert sum(src_counts.values()) == 900
    assert len(src_counts) == 8


def test_7_license_totals():
    """Test 7: License distribution matches verified totals (740 MIT, 100 Apache-2.0, 60 Llama-2)."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    lic_counts: dict[str, int] = {}
    for item in train_items:
        lic = item["metadata"].get("license", "unknown")
        lic_counts[lic] = lic_counts.get(lic, 0) + 1

    assert lic_counts.get("MIT") == 740
    assert lic_counts.get("Apache-2.0") == 100
    assert lic_counts.get("Llama-2") == 60


def test_8_security_authorization_metadata():
    """Test 8: Security authorization metadata totals 900 across valid statuses."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    auth_counts: dict[str, int] = {}
    for item in train_items:
        auth = item["metadata"].get("authorization", "not_applicable")
        auth_counts[auth] = auth_counts.get(auth, 0) + 1

    assert sum(auth_counts.values()) == 900
    assert auth_counts.get("defensive") == 477
    assert auth_counts.get("authorized") == 207


def test_9_training_validation_separation():
    """Test 9: Training and validation sets have zero overlapping user prompts."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    val_items = load_jsonl(VAL_V030_PATH)

    train_prompts = {
        m.get("content", "").strip().lower()
        for item in train_items
        for m in item.get("messages", [])
        if m.get("role") == "user"
    }

    val_dups = [
        m.get("content", "").strip().lower()
        for item in val_items
        for m in item.get("messages", [])
        if m.get("role") == "user" and m.get("content", "").strip().lower() in train_prompts
    ]

    assert len(val_dups) == 0, f"Training/validation overlap found: {val_dups}"


def test_10_training_evaluation_separation():
    """Test 10: Training and evaluation sets have zero overlapping user prompts."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    eval_items = load_jsonl(SECBENCH_EVAL_PATH) + load_jsonl(SWEBENCH_EVAL_PATH)

    train_prompts = {
        m.get("content", "").strip().lower()
        for item in train_items
        for m in item.get("messages", [])
        if m.get("role") == "user"
    }

    eval_dups = [
        m.get("content", "").strip().lower()
        for item in eval_items
        for m in item.get("messages", [])
        if m.get("role") == "user" and m.get("content", "").strip().lower() in train_prompts
    ]

    assert len(eval_dups) == 0, f"Training/evaluation contamination found: {eval_dups}"


def test_11_schema_validity_all_rows():
    """Test 11: Schema validation succeeds across all 900 training and 90 validation rows."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    val_items = load_jsonl(VAL_V030_PATH)

    for item in train_items + val_items:
        TrainingExample.model_validate(item)


def test_12_no_stale_200_example_capability_report():
    """Test 12: Capability gap script analyzes 900 items instead of 200 items by default."""
    gap_result = analyze_gaps(TRAIN_V030_PATH, target_total=1000)
    assert gap_result["total_examples"] == 900


def test_13_no_unknown_license_in_training():
    """Test 13: No unknown/unverified license in training candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        lic = item["metadata"].get("license")
        assert lic in ("MIT", "Apache-2.0", "Llama-2")


def test_14_provenance_integrity():
    """Test 14: Every training candidate has source and source_details."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        meta = item["metadata"]
        assert meta.get("source") is not None
        assert meta.get("source_details") is not None


def test_15_no_azure_jobs_launched():
    """Test 15: No Azure training jobs triggered."""
    runs_dir = DATA_DIR / "metadata" / "training_runs"
    if runs_dir.exists():
        for run_file in runs_dir.glob("*.json"):
            content = run_file.read_text(encoding="utf-8")
            assert "azure" not in content.lower()
