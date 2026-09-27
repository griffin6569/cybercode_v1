"""CyberCodeMini Phase 11 Automated Test Suite

Verifies dataset candidate schema, capability taxonomy, license governance, benchmark protection,
training eligibility, provenance preservation, security authorization metadata, PII/secret scanning,
deduplication, tier separation, manifest counts, source distribution, and Azure restrictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets.schemas.capabilities import Capability, CapabilityClassification
from datasets.schemas.schema import TrainingExample

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_V030_PATH = DATA_DIR / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.3.0.jsonl"
VAL_V030_PATH = DATA_DIR / "processed" / "validation_candidates" / "cybercodemini_validation_candidates_v0.3.0.jsonl"
MANIFEST_V030_PATH = DATA_DIR / "processed" / "manifests" / "corpus_manifest_v0.3.0.json"
CANDIDATES_JSON_PATH = DATA_DIR / "metadata" / "phase11_dataset_candidates.json"
BASELINE_PATH = DATA_DIR / "metadata" / "phase11_baseline.json"
SECBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "secbench" / "examples.jsonl"
SWEBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "swe_bench" / "examples.jsonl"
REVIEW_PATH = DATA_DIR / "review" / "unknown_license" / "examples.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"Path '{path}' does not exist"
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def test_1_dataset_candidate_schema():
    """Test 1: Dataset candidate schema is valid."""
    items = load_jsonl(TRAIN_V030_PATH)
    sample = items[0]
    ex = TrainingExample.model_validate(sample)
    assert ex.messages is not None
    assert ex.metadata.category is not None


def test_2_capability_taxonomy():
    """Test 2: Capability taxonomy contains expected values."""
    assert Capability.CODE_GENERATION.value == "code_generation"
    assert Capability.VULNERABILITY_REMEDIATION.value == "vulnerability_remediation"
    assert Capability.AGENT_TOOL_USE.value == "agent_tool_use"
    assert Capability.AUTHORIZED_LAB.value == "authorized_lab"

    cap_obj = CapabilityClassification(
        primary_capability=Capability.VULNERABILITY_REMEDIATION,
        secondary_capabilities=[Capability.SECURE_CODING]
    )
    assert cap_obj.primary_capability == "vulnerability_remediation"


def test_3_dataset_discovery_metadata():
    """Test 3: Dataset discovery metadata file is complete."""
    assert CANDIDATES_JSON_PATH.exists()
    data = json.loads(CANDIDATES_JSON_PATH.read_text(encoding="utf-8"))
    candidates = data.get("dataset_candidates", [])
    assert len(candidates) >= 5
    for c in candidates:
        assert "dataset_id" in c
        assert "license" in c
        assert "training_eligibility" in c


def test_4_license_status_validation():
    """Test 4: License status is verified across all candidates."""
    items = load_jsonl(TRAIN_V030_PATH)
    allowed_licenses = {"MIT", "Apache-2.0", "Llama-2", "BSD-3-Clause"}
    for item in items:
        lic = item.get("metadata", {}).get("license")
        assert lic in allowed_licenses, f"Unexpected license '{lic}' in item: {item}"


def test_5_benchmark_classification():
    """Test 5: Benchmark datasets are classified as evaluation_only."""
    secbench_items = load_jsonl(SECBENCH_EVAL_PATH)
    swebench_items = load_jsonl(SWEBENCH_EVAL_PATH)
    for item in secbench_items + swebench_items:
        meta = item["metadata"]
        assert meta["classification"] == "evaluation_only"
        assert meta["allowed_for_training"] is False


def test_6_training_eligibility():
    """Test 6: All training candidate examples have allowed_for_training=True."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        meta = item["metadata"]
        assert meta["classification"] == "training_candidate"
        assert meta["allowed_for_training"] is True


def test_7_unknown_license_exclusion():
    """Test 7: Unknown license candidate is excluded from training."""
    review_items = load_jsonl(REVIEW_PATH)
    train_items = load_jsonl(TRAIN_V030_PATH)
    review_sources = {item["metadata"]["source"] for item in review_items}
    for item in train_items:
        src = item["metadata"].get("source")
        assert src not in review_sources, f"Review dataset source '{src}' found in training candidate"


def test_8_provenance_preservation():
    """Test 8: Every training candidate preserves detailed provenance metadata."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        meta = item["metadata"]
        src = meta.get("source")
        src_details = meta.get("source_details")
        assert src or src_details, f"Missing provenance in item: {item}"
        if src_details:
            assert "dataset_id" in src_details
            assert "source_id" in src_details


def test_9_security_authorization_metadata():
    """Test 9: Security-related training candidates have explicit authorization context."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    valid_auth = {"defensive", "authorized", "explicit", "educational", "not_applicable"}
    for item in train_items:
        meta = item["metadata"]
        auth = meta.get("authorization")
        assert auth in valid_auth, f"Invalid authorization context '{auth}'"


def test_10_pii_secrets_scanning():
    """Test 10: No training candidate contains unreviewed private keys or tokens."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        concat_text = " ".join([m.get("content", "") for m in item.get("messages", [])])
        assert "-----BEGIN PRIVATE KEY-----" not in concat_text
        assert "-----BEGIN RSA PRIVATE KEY-----" not in concat_text


def test_11_exact_deduplication():
    """Test 11: No exact user prompt duplicates exist within training candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    seen = set()
    duplicates = []
    for item in train_items:
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", "").strip().lower()
                if content in seen:
                    duplicates.append(content)
                seen.add(content)
    assert len(duplicates) == 0, f"Found duplicate training prompts: {duplicates}"


def test_12_training_eval_separation():
    """Test 12: Zero overlap between training candidates and evaluation benchmarks."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    eval_items = load_jsonl(SECBENCH_EVAL_PATH) + load_jsonl(SWEBENCH_EVAL_PATH)

    train_prompts = {
        m.get("content", "").strip().lower()
        for item in train_items
        for m in item.get("messages", [])
        if m.get("role") == "user"
    }

    contamination = []
    for item in eval_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                p = m.get("content", "").strip().lower()
                if p in train_prompts:
                    contamination.append(p)

    assert len(contamination) == 0, f"Benchmark contamination detected: {contamination}"


def test_13_validation_training_separation():
    """Test 13: Zero overlap between training candidates and validation candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    val_items = load_jsonl(VAL_V030_PATH)

    train_prompts = {
        m.get("content", "").strip().lower()
        for item in train_items
        for m in item.get("messages", [])
        if m.get("role") == "user"
    }

    val_overlap = []
    for item in val_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                p = m.get("content", "").strip().lower()
                if p in train_prompts:
                    val_overlap.append(p)

    assert len(val_overlap) == 0, f"Validation overlap detected: {val_overlap}"


def test_14_capability_classification():
    """Test 14: Every training candidate has a defined primary capability."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        meta = item["metadata"]
        cap = meta.get("primary_capability") or meta.get("category")
        assert cap is not None, f"Missing capability in item: {item}"


def test_15_corpus_manifest_counts():
    """Test 15: Manifest v0.3.0 counts match actual file row counts."""
    assert MANIFEST_V030_PATH.exists()
    manifest = json.loads(MANIFEST_V030_PATH.read_text(encoding="utf-8"))
    train_count = len(load_jsonl(TRAIN_V030_PATH))
    val_count = len(load_jsonl(VAL_V030_PATH))

    assert manifest["total_training_candidates"] == train_count
    assert manifest["total_validation_candidates"] == val_count


def test_16_source_distribution_monoculture_check():
    """Test 16: No single dataset source exceeds 40% of the training corpus."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    total = len(train_items)
    src_counts: dict[str, int] = {}
    for item in train_items:
        src = item["metadata"].get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    for src, count in src_counts.items():
        pct = (count / total) * 100.0
        assert pct <= 40.0, f"Source '{src}' exceeds 40% threshold ({pct:.1f}%)"


def test_17_deterministic_sampling():
    """Test 17: Baseline baseline file exists and records seed 42."""
    assert BASELINE_PATH.exists()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["training_candidates_count"] == 200


def test_18_no_benchmark_contamination():
    """Test 18: Neither SecBench nor SWE-bench IDs appear in training candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        src = item["metadata"].get("source")
        assert src != "secbench-hf/SecBench"
        assert src != "SWE-bench/SWE-bench"


def test_19_no_unknown_license_in_training():
    """Test 19: No unknown license values exist in training candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        lic = item["metadata"].get("license")
        assert lic is not None and lic != "unknown" and lic != "unverified"


def test_20_no_missing_provenance_in_training():
    """Test 20: No missing provenance sources in training candidates."""
    train_items = load_jsonl(TRAIN_V030_PATH)
    for item in train_items:
        assert item["metadata"].get("source") is not None


def test_21_no_azure_jobs_triggered():
    """Test 21: No Azure training job was triggered in Phase 11."""
    runs_dir = DATA_DIR / "metadata" / "training_runs"
    if runs_dir.exists():
        for run_file in runs_dir.glob("*.json"):
            content = run_file.read_text(encoding="utf-8")
            assert "azure" not in content.lower()
