"""CyberCodeMini Phase 12 Automated Test Suite

Verifies frozen corpus immutability, checksum verification, exact row counts, schema validity,
duplicate detection, manifest consistency, training config consistency, tokenization pre-flight,
evaluation isolation, reproducibility metadata, and Azure safety controls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cybercode_datasets.schemas.schema import TrainingExample

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FROZEN_DIR = DATA_DIR / "frozen" / "v0.3.0"
FROZEN_TRAIN_PATH = FROZEN_DIR / "training.jsonl"
FROZEN_VAL_PATH = FROZEN_DIR / "validation.jsonl"
FROZEN_MANIFEST_PATH = FROZEN_DIR / "manifest.json"
FROZEN_README_PATH = FROZEN_DIR / "README.md"
SHA256SUMS_PATH = FROZEN_DIR / "SHA256SUMS.txt"

FROZEN_MANIFEST_META_PATH = DATA_DIR / "metadata" / "frozen_corpus_manifest_v0.3.0.json"
AGENT_REPORT_PATH = DATA_DIR / "metadata" / "frozen_agent_capability_report_v0.3.0.json"
CONFIG_FROZEN_PATH = DATA_DIR / "metadata" / "training_config_frozen_v0.3.0.json"
REPRODUCIBILITY_PATH = DATA_DIR / "metadata" / "reproducibility_v0.3.0.json"

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


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_1_frozen_files_exist():
    """Test 1: Frozen files and directory structure exist."""
    assert FROZEN_DIR.exists()
    assert FROZEN_TRAIN_PATH.exists()
    assert FROZEN_VAL_PATH.exists()
    assert FROZEN_MANIFEST_PATH.exists()
    assert FROZEN_README_PATH.exists()
    assert SHA256SUMS_PATH.exists()


def test_2_checksum_verification():
    """Test 2: Computed SHA-256 checksums match SHA256SUMS.txt and manifest."""
    train_sha = compute_sha256(FROZEN_TRAIN_PATH)
    val_sha = compute_sha256(FROZEN_VAL_PATH)

    sums_text = SHA256SUMS_PATH.read_text(encoding="utf-8")
    assert train_sha in sums_text
    assert val_sha in sums_text

    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["checksums"]["training_sha256"] == train_sha
    assert manifest["checksums"]["validation_sha256"] == val_sha


def test_3_exact_row_counts():
    """Test 3: Row counts match expectations (900 train, 90 val, 100 secbench, 100 swebench, 50 review)."""
    train_rows = len(load_jsonl(FROZEN_TRAIN_PATH))
    val_rows = len(load_jsonl(FROZEN_VAL_PATH))
    secbench_rows = len(load_jsonl(SECBENCH_EVAL_PATH))
    swebench_rows = len(load_jsonl(SWEBENCH_EVAL_PATH))
    review_rows = len(load_jsonl(REVIEW_PATH))

    assert train_rows == 900
    assert val_rows == 90
    assert secbench_rows == 100
    assert swebench_rows == 100
    assert review_rows == 50


def test_4_schema_validity_frozen():
    """Test 4: Every row in frozen training and validation JSONL is schema valid."""
    train_items = load_jsonl(FROZEN_TRAIN_PATH)
    val_items = load_jsonl(FROZEN_VAL_PATH)

    for item in train_items + val_items:
        ex = TrainingExample.model_validate(item)
        assert ex.messages is not None
        assert ex.metadata.category is not None


def test_5_duplicate_detection_frozen():
    """Test 5: Zero exact duplicates between frozen training, validation, and evaluation benchmarks."""
    train_items = load_jsonl(FROZEN_TRAIN_PATH)
    val_items = load_jsonl(FROZEN_VAL_PATH)
    eval_items = load_jsonl(SECBENCH_EVAL_PATH) + load_jsonl(SWEBENCH_EVAL_PATH)

    train_prompts = {
        m.get("content", "").strip().lower()
        for item in train_items
        for m in item.get("messages", [])
        if m.get("role") == "user"
    }

    for item in val_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                p = m.get("content", "").strip().lower()
                assert p not in train_prompts, f"Training/validation duplicate found: {p}"

    for item in eval_items:
        for m in item.get("messages", []):
            if m.get("role") == "user":
                p = m.get("content", "").strip().lower()
                assert p not in train_prompts, f"Training/evaluation contamination found: {p}"


def test_6_manifest_consistency():
    """Test 6: Frozen manifest and metadata manifest match."""
    m1 = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    m2 = json.loads(FROZEN_MANIFEST_META_PATH.read_text(encoding="utf-8"))
    assert m1["checksums"]["training_sha256"] == m2["checksums"]["training_sha256"]
    assert m1["row_counts"]["training"] == 900
    assert m1["row_counts"]["validation"] == 90


def test_7_training_config_consistency():
    """Test 7: Frozen training configuration matches project settings."""
    assert CONFIG_FROZEN_PATH.exists()
    cfg = json.loads(CONFIG_FROZEN_PATH.read_text(encoding="utf-8"))
    assert cfg["base_model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg["lora_parameters"]["r"] == 16
    assert cfg["lora_parameters"]["alpha"] == 32
    assert cfg["hyperparameters"]["max_seq_length"] == 2048


def test_8_frozen_corpus_immutability():
    """Test 8: Frozen training JSONL matches v0.3.0 training candidates file byte-for-byte."""
    v030_train_path = DATA_DIR / "processed" / "training_candidates" / "cybercodemini_train_candidates_v0.3.0.jsonl"
    sha1 = compute_sha256(FROZEN_TRAIN_PATH)
    sha2 = compute_sha256(v030_train_path)
    assert sha1 == sha2, "Frozen training JSONL does not match v0.3.0 training candidate file"


def test_9_agent_capability_report():
    """Test 9: Agent capability report records 147 full trajectories."""
    assert AGENT_REPORT_PATH.exists()
    report = json.loads(AGENT_REPORT_PATH.read_text(encoding="utf-8"))
    assert report["total_agent_examples"] == 147
    assert report["structural_tiers"]["full_agent_trajectory"] == 147


def test_10_evaluation_isolation():
    """Test 10: Benchmark dataset sources do not exist in frozen training or validation."""
    train_items = load_jsonl(FROZEN_TRAIN_PATH)
    val_items = load_jsonl(FROZEN_VAL_PATH)

    for item in train_items + val_items:
        src = item["metadata"].get("source")
        assert src != "secbench-hf/SecBench"
        assert src != "SWE-bench/SWE-bench"


def test_11_reproducibility_metadata():
    """Test 11: Reproducibility metadata file is complete."""
    assert REPRODUCIBILITY_PATH.exists()
    repro = json.loads(REPRODUCIBILITY_PATH.read_text(encoding="utf-8"))
    assert repro["random_seed"] == 42
    assert "training_sha256" in repro["dataset_checksums"]


def test_12_no_azure_jobs_launched():
    """Test 12: No Azure training job was launched in Phase 12."""
    runs_dir = DATA_DIR / "metadata" / "training_runs"
    if runs_dir.exists():
        for run_file in runs_dir.glob("*.json"):
            content = run_file.read_text(encoding="utf-8")
            assert "azure" not in content.lower()
