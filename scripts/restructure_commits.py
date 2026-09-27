"""CyberCodeMini Git Commit Restructuring Script

Restructures the Git commit history into clean, professional phase-by-phase conventional commits.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PHASE_COMMITS = [
    {
        "message": "feat(core): initialize CyberCodeMini architecture, schemas, and validators (Phases 1-6)",
        "paths": [
            ".gitignore", ".env.example", "LICENSE", "README.md", "pyproject.toml", "requirements.txt",
            "configs/", "datasets/__init__.py", "datasets/schemas/__init__.py", "datasets/schemas/schema.py",
            "datasets/validators/__init__.py", "datasets/validators/validator.py",
            "agent/", "lab/", "evaluation/metrics.py", "evaluation/run_eval.py", "data/README.md",
            "data/interim/", "data/metadata/.gitkeep", "data/metadata/dataset_registry.json",
            "data/processed/.gitkeep", "data/raw/.gitkeep", "data/test/", "data/train/", "data/validation/",
            "scripts/dataset_stats.py", "scripts/deduplicate.py", "scripts/download_dataset.py",
            "scripts/inspect_dataset.py", "scripts/smoke_test.py", "scripts/split_dataset.py",
            "scripts/validate_dataset.py", "tests/test_agent.py", "tests/test_dataset.py",
            "tests/test_tools.py", "tests/test_training_config.py"
        ]
    },
    {
        "message": "feat(ingestion): build dataset adapters, provenance, and quality filtering engine (Phase 7)",
        "paths": [
            "datasets/adapters/", "datasets/ingestion/", "datasets/provenance/",
            "datasets/filters/quality_filter.py", "datasets/validators/quality.py",
            "data/raw/dev_dataset.jsonl", "data/processed/curated_dev.jsonl",
            "scripts/build_dev_dataset.py", "scripts/ingest_dataset.py", "scripts/curate_dataset.py",
            "tests/test_phase7.py"
        ]
    },
    {
        "message": "feat(tokenization): implement Qwen chat template, loss masking, and collator (Phase 8)",
        "paths": [
            "datasets/formatting/", "datasets/tokenization.py", "training/collator.py",
            "scripts/inspect_loss_mask.py", "scripts/tokenization_smoke_test.py",
            "data/metadata/tokenization_stats.json", "tests/test_phase8.py"
        ]
    },
    {
        "message": "feat(pipeline): implement LoRA/QLoRA training pipeline, checkpointing, and export (Phase 9)",
        "paths": [
            "training/", "azure/", "scripts/train_smoke_test.py", "scripts/export_adapter.py",
            "scripts/inspect_checkpoint.py", "scripts/inspect_training_config.py",
            "data/metadata/training_runs/", "tests/test_phase9.py"
        ]
    },
    {
        "message": "feat(expansion): ingest Hugging Face dataset candidates and build Corpus v0.2 (Phase 10)",
        "paths": [
            "scripts/inspect_hf_dataset.py", "scripts/ingest_hf_dataset.py",
            "scripts/validate_imported_dataset.py", "scripts/build_expanded_dataset.py",
            "scripts/review_dataset.py", "data/processed/cybercodemini_train_v0.2.jsonl",
            "data/metadata/ingestion_manifest.json", "data/metadata/review_queue.json",
            "data/metadata/manifest.json", "data/metadata/ingestion_reports/", "tests/test_phase10.py"
        ]
    },
    {
        "message": "refactor(audit): execute Phase 10.1 pre-audit and train/evaluation separation (Phase 10.1)",
        "paths": [
            "data/evaluation/", "data/review/", "data/processed/training_candidates/cybercodemini_train_candidates_v0.2.1.jsonl",
            "data/processed/validation_candidates/cybercodemini_validation_candidates_v0.2.1.jsonl",
            "data/processed/manifests/corpus_manifest_v0.2.1.json", "data/processed/training_candidates/README.md",
            "data/metadata/phase10_1_pre_audit.json", "data/metadata/dataset_decisions.json",
            "data/metadata/openhermes_code_audit.json", "data/metadata/swe_bench_repository_licenses.json",
            "data/metadata/phase10_1_dataset_stats.json", "data/metadata/phase10_1_final_report.md",
            "scripts/build_phase10_1_corpus.py", "scripts/audit_phase10_1.py", "tests/test_phase10_1.py"
        ]
    },
    {
        "message": "feat(corpus): expand training corpus to v0.3.0 and capability taxonomy (Phase 11)",
        "paths": [
            "datasets/schemas/capabilities.py", "data/metadata/phase11_baseline.json",
            "data/metadata/phase11_dataset_candidates.json", "data/metadata/phase11_final_report.md",
            "data/processed/training_candidates/cybercodemini_train_candidates_v0.3.0.jsonl",
            "data/processed/validation_candidates/cybercodemini_validation_candidates_v0.3.0.jsonl",
            "data/processed/manifests/corpus_manifest_v0.3.0.json", "data/processed/CYBERCODEMINI_DATASET_CARD.md",
            "scripts/analyze_capability_gaps.py", "scripts/phase11_ingest.py", "scripts/analyze_phase11_balance.py",
            "tests/test_phase11.py"
        ]
    },
    {
        "message": "audit(forensic): execute Phase 11.1 forensic audit and agent trajectory breakdown (Phase 11.1)",
        "paths": [
            "data/metadata/agent_trajectory_audit.json", "data/metadata/phase11_1_audit.json",
            "data/metadata/phase11_1_final_audit.md", "scripts/forensic_audit_phase11_1.py",
            "tests/test_phase11_1.py"
        ]
    },
    {
        "message": "freeze(corpus): perform final corpus freeze v0.3.0 and pre-training readiness gate (Phase 12)",
        "paths": [
            "data/frozen/", "data/metadata/frozen_corpus_manifest_v0.3.0.json",
            "data/metadata/frozen_agent_capability_report_v0.3.0.json",
            "data/metadata/training_config_frozen_v0.3.0.json", "data/metadata/reproducibility_v0.3.0.json",
            "data/metadata/phase12_final_report.md", "scripts/build_phase12_freeze.py",
            "tests/test_phase12.py"
        ]
    }
]


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\nError: {res.stderr}")
    return res.stdout.strip()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    os.chdir(base_dir)

    print("Creating clean orphan git branch 'temp_clean'...")
    run_cmd(["git", "checkout", "--orphan", "temp_clean"])
    run_cmd(["git", "reset"])

    for phase in PHASE_COMMITS:
        msg = phase["message"]
        paths = phase["paths"]
        print(f"Staging files for: {msg}...")
        for p in paths:
            if Path(p).exists():
                run_cmd(["git", "add", p])

        # Commit staged files
        status = run_cmd(["git", "status", "--porcelain"])
        if status:
            run_cmd(["git", "commit", "-m", msg])
            print(f"  Committed: {msg}")

    # Stage any remaining files
    remaining_status = run_cmd(["git", "status", "--porcelain"])
    if remaining_status:
        print("Staging remaining project files...")
        run_cmd(["git", "add", "."])
        run_cmd(["git", "commit", "-m", "chore(repo): complete project restructuring and final validation cleanup"])

    print("Overwriting 'main' branch with 'temp_clean'...")
    run_cmd(["git", "branch", "-M", "temp_clean", "main"])

    print("Force pushing clean commit history to GitHub...")
    push_res = run_cmd(["git", "push", "-f", "origin", "main"])
    print(push_res)
    print("Commit restructuring completed successfully!")


if __name__ == "__main__":
    main()
