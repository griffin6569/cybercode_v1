"""CyberCodeMini Phase 10.1 Automated Test Suite

Tests all 12 hard requirements for dataset audit, separation, governance, provenance,
benchmark contamination prevention, and Azure constraints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_CANDIDATES_PATH = DATA_DIR / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.2.1.jsonl"
VAL_CANDIDATES_PATH = DATA_DIR / "processed" / "validation_candidates" / "cybercodemini_validation_candidates_v0.2.1.jsonl"
SECBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "secbench" / "examples.jsonl"
SECBENCH_MANIFEST_PATH = DATA_DIR / "evaluation" / "secbench" / "manifest.json"
SWEBENCH_EVAL_PATH = DATA_DIR / "evaluation" / "swe_bench" / "examples.jsonl"
SWEBENCH_MANIFEST_PATH = DATA_DIR / "evaluation" / "swe_bench" / "manifest.json"
REVIEW_PATH = DATA_DIR / "review" / "unknown_license" / "examples.jsonl"
CORPUS_MANIFEST_PATH = DATA_DIR / "processed" / "manifests" / "corpus_manifest_v0.2.1.json"
DATASET_DECISIONS_PATH = DATA_DIR / "metadata" / "dataset_decisions.json"
RAW_DEV_DATASET_PATH = DATA_DIR / "raw" / "dev_dataset.jsonl"
HISTORICAL_V02_PATH = DATA_DIR / "processed" / "cybercodemini_train_v0.2.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    items = []
    assert path.exists(), f"Path {path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def test_1_secbench_not_in_training():
    """Test 1: SecBench examples cannot appear in training candidates."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    for item in train_items:
        meta = item.get("metadata", {})
        src = meta.get("source")
        src_details = meta.get("source_details", {})
        dataset_id = src_details.get("dataset_id") if isinstance(src_details, dict) else None
        assert src != "secbench-hf/SecBench", f"SecBench found in training candidate: {item}"
        assert dataset_id != "secbench-hf/SecBench", f"SecBench found in source_details of training candidate: {item}"


def test_2_swebench_not_in_training():
    """Test 2: SWE-bench benchmark examples cannot appear in training candidates."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    for item in train_items:
        meta = item.get("metadata", {})
        src = meta.get("source")
        src_details = meta.get("source_details", {})
        dataset_id = src_details.get("dataset_id") if isinstance(src_details, dict) else None
        assert src != "SWE-bench/SWE-bench", f"SWE-bench found in training candidate: {item}"
        assert dataset_id != "SWE-bench/SWE-bench", f"SWE-bench found in source_details of training candidate: {item}"


def test_3_unknown_license_not_in_training():
    """Test 3: Unknown-license examples cannot appear in training candidates."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    for item in train_items:
        meta = item.get("metadata", {})
        lic = meta.get("license")
        assert lic is not None and lic != "unknown" and lic != "unverified", f"Unknown/unverified license in training candidate: {item}"


def test_4_every_training_candidate_has_provenance():
    """Test 4: Every training candidate has provenance metadata."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    for item in train_items:
        meta = item.get("metadata", {})
        src = meta.get("source")
        src_details = meta.get("source_details")
        assert src or src_details, f"Training candidate missing provenance source: {item}"
        if src_details:
            assert isinstance(src_details, dict)
            assert "dataset_id" in src_details, f"Missing dataset_id in source_details: {src_details}"


def test_5_every_training_candidate_valid_classification():
    """Test 5: Every training candidate has a valid classification."""
    valid_classifications = {"training_candidate", "validation_candidate", "evaluation_only", "review", "excluded"}
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    val_items = load_jsonl(VAL_CANDIDATES_PATH)

    for item in train_items:
        meta = item.get("metadata", {})
        cls_val = meta.get("classification")
        assert cls_val == "training_candidate", f"Expected training_candidate, got '{cls_val}'"

    for item in val_items:
        meta = item.get("metadata", {})
        cls_val = meta.get("classification")
        assert cls_val == "validation_candidate", f"Expected validation_candidate, got '{cls_val}'"


def test_6_no_training_eval_exact_duplicates():
    """Test 6: Training/evaluation exact duplicates are zero."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    secbench_items = load_jsonl(SECBENCH_EVAL_PATH)
    swebench_items = load_jsonl(SWEBENCH_EVAL_PATH)

    train_user_prompts = set()
    for item in train_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                train_user_prompts.add(m.get("content", "").strip().lower())

    eval_items = secbench_items + swebench_items
    duplicates = []
    for item in eval_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                prompt = m.get("content", "").strip().lower()
                if prompt in train_user_prompts:
                    duplicates.append(prompt)

    assert len(duplicates) == 0, f"Found {len(duplicates)} exact duplicates between training and evaluation: {duplicates}"


def test_7_dataset_decisions_contain_all_phase10_datasets():
    """Test 7: Dataset decisions contain all Phase 10 datasets."""
    assert DATASET_DECISIONS_PATH.exists()
    decisions = json.loads(DATASET_DECISIONS_PATH.read_text(encoding="utf-8"))
    datasets = decisions.get("datasets", [])
    dataset_ids = {d["dataset_id"] for d in datasets}

    expected = {
        "cybercode_curated_dev",
        "secbench-hf/SecBench",
        "SWE-bench/SWE-bench",
        "OpenHermes-Code-Sample",
        "candidate_unknown_license_db",
    }
    assert expected.issubset(dataset_ids), f"Missing datasets in decisions: {expected - dataset_ids}"


def test_8_raw_evidence_not_deleted():
    """Test 8: Original/raw evidence was not silently deleted."""
    assert RAW_DEV_DATASET_PATH.exists(), "Raw dev dataset dev_dataset.jsonl was deleted"
    assert HISTORICAL_V02_PATH.exists(), "Historical Phase 10 v0.2 file cybercodemini_train_v0.2.jsonl was deleted"
    
    raw_count = len(load_jsonl(RAW_DEV_DATASET_PATH))
    assert raw_count == 120, f"Expected 120 raw dev examples, found {raw_count}"

    v02_count = len(load_jsonl(HISTORICAL_V02_PATH))
    assert v02_count == 420, f"Expected 420 historical v0.2 examples, found {v02_count}"


def test_9_training_manifest_count_equals_actual_rows():
    """Test 9: The training manifest count equals the actual number of rows."""
    assert CORPUS_MANIFEST_PATH.exists()
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    train_rows = len(load_jsonl(TRAIN_CANDIDATES_PATH))
    val_rows = len(load_jsonl(VAL_CANDIDATES_PATH))

    assert manifest["training_candidate_count"] == train_rows
    assert manifest["validation_candidate_count"] == val_rows


def test_10_evaluation_manifest_counts_equal_actual_rows():
    """Test 10: Evaluation manifest counts equal actual evaluation rows."""
    assert SECBENCH_MANIFEST_PATH.exists()
    secbench_manifest = json.loads(SECBENCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    secbench_rows = len(load_jsonl(SECBENCH_EVAL_PATH))
    assert secbench_manifest["sample_count"] == secbench_rows

    assert SWEBENCH_MANIFEST_PATH.exists()
    swebench_manifest = json.loads(SWEBENCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    swebench_rows = len(load_jsonl(SWEBENCH_EVAL_PATH))
    assert swebench_manifest["sample_count"] == swebench_rows


def test_11_no_conflicting_classification_metadata():
    """Test 11: No example has conflicting classification metadata."""
    train_items = load_jsonl(TRAIN_CANDIDATES_PATH)
    val_items = load_jsonl(VAL_CANDIDATES_PATH)
    secbench_items = load_jsonl(SECBENCH_EVAL_PATH)
    swebench_items = load_jsonl(SWEBENCH_EVAL_PATH)
    review_items = load_jsonl(REVIEW_PATH)

    for item in train_items:
        meta = item["metadata"]
        assert meta["classification"] == "training_candidate"
        assert meta["allowed_for_training"] is True

    for item in val_items:
        meta = item["metadata"]
        assert meta["classification"] == "validation_candidate"
        assert meta["allowed_for_training"] is True

    for item in secbench_items + swebench_items:
        meta = item["metadata"]
        assert meta["classification"] == "evaluation_only"
        assert meta["allowed_for_training"] is False

    for item in review_items:
        meta = item["metadata"]
        assert meta["classification"] == "review"
        assert meta["allowed_for_training"] is False


def test_12_no_azure_jobs_launched():
    """Test 12: No Azure-related training job was launched by this phase."""
    runs_dir = DATA_DIR / "metadata" / "training_runs"
    if runs_dir.exists():
        for run_file in runs_dir.glob("*.json"):
            content = run_file.read_text(encoding="utf-8")
            assert "azure" not in content.lower(), f"Azure training run detected in {run_file}: {content}"
