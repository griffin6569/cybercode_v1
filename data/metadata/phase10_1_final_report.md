# CyberCodeMini Phase 10.1 — Dataset Audit & Train/Evaluation Separation Final Report

**Date**: 2026-09-27  
**Phase**: Phase 10.1  
**Corpus Version**: v0.2.1  
**Status**: READY FOR NEXT DATASET EXPANSION  

---

## 1. Executive Summary

Phase 10.1 performed a comprehensive audit and reorganization of the CyberCodeMini dataset corpus produced in Phase 10. The goal was to eliminate benchmark contamination, enforce strict physical separation between training candidates and evaluation benchmarks, verify dataset and underlying content licensing, and establish machine-readable provenance across all data tiers prior to any model fine-tuning.

Key outcomes:
- Benchmark datasets (**SecBench** and **SWE-bench**) were removed from training candidates and placed in dedicated evaluation directories.
- Data tiers were restructured under `data/processed/training_candidates/`, `data/processed/validation_candidates/`, `data/evaluation/`, and `data/review/`.
- Permissive dataset-level licenses were disentangled from underlying repository content rights.
- Unknown-license candidates remained isolated in `data/review/unknown_license/`.
- All 94 automated test assertions passed with zero training/evaluation contamination.

---

## 2. Dataset Inventory

| Dataset Identifier | Raw Source / URL | Audited Role | Total Sample Count |
| :--- | :--- | :--- | :--- |
| `cybercode_curated_dev` | `data/raw/dev_dataset.jsonl` | Curated Internal Dev Set | 120 |
| `secbench-hf/SecBench` | `https://huggingface.co/datasets/secbench-hf/SecBench` | Security Review Evaluation | 100 |
| `SWE-bench/SWE-bench` | `https://huggingface.co/datasets/SWE-bench/SWE-bench` | Agent Trajectory Evaluation | 100 |
| `OpenHermes-Code-Sample` | `https://huggingface.co/datasets/teknium/openhermes` | Code Instruction Tuning | 100 |
| `candidate_unknown_license_db` | `https://example.org/raw_dump` | Unverified License Review Queue | 50 |
| **Total Inspected** | | | **470** |

---

## 3. License Audit

| Dataset | Declared License | License Verification | Underlying Content Considerations | Training Eligibility |
| :--- | :--- | :--- | :--- | :--- |
| `cybercode_curated_dev` | MIT | Verified | Synthetically generated / internally curated security code | `training_candidate` / `validation_candidate` |
| `secbench-hf/SecBench` | MIT | Verified | Real-world vulnerability & fix pairs (Benchmark) | `evaluation_only` |
| `SWE-bench/SWE-bench` | MIT | Verified | Aggregates 12+ GitHub repos (BSD/MIT/Apache) | `evaluation_only` |
| `OpenHermes-Code-Sample` | Apache-2.0 | Verified | Open instruction tuning code samples | `training_candidate` |
| `candidate_unknown_license_db` | Unknown | Unverified | Raw code dump with no license header | `review` (Excluded from training) |

*Notice: In accordance with CyberCodeMini engineering and data-governance policy, a permissive dataset license does not automatically grant approval for model training on all underlying artifacts. Benchmark status, license verification, authorization context, and provenance are evaluated as independent dimensions.*

---

## 4. Benchmark Audit

### SecBench (`secbench-hf/SecBench`)
- **Reason for Separation**: SecBench contains standard security review and vulnerability remediation evaluation benchmarks. Including SecBench in the ordinary training corpus distorts model evaluation results.
- **Classification**: `evaluation_only`
- **Location**: `data/evaluation/secbench/` (100 items)

### SWE-bench (`SWE-bench/SWE-bench`)
- **Reason for Separation**: SWE-bench is a premier benchmark for evaluating coding agents on full repository tasks. Training on SWE-bench instances causes benchmark leakage and invalidates agent capability evaluations.
- **Classification**: `evaluation_only`
- **Location**: `data/evaluation/swe_bench/` (100 items)

---

## 5. OpenHermes Audit

- **HF Identifier**: `teknium/openhermes` (revision `main`)
- **License**: Apache-2.0 (Verified via dataset card)
- **Content**: General programming instructions and code structure generations.
- **Audit Finding**: Dataset license is verified and content contains non-benchmark programming instruction samples. Cleared for training candidates.
- **Classification**: `training_candidate` (100 items)

---

## 6. Unknown-License Candidate

- **Identifier**: `candidate_unknown_license_db`
- **Audit Finding**: License metadata is unverified/unknown and upstream repository provenance cannot be established.
- **Decision**: Maintained in `data/review/unknown_license/` (50 items). Strictly prohibited from entering training candidates until license clearance is obtained.

---

## 7. Final Corpus Accounting

```
Original Phase 10 v0.2 size:      420
Phase 10 Review items:             50
Total Inspected Corpus:           470

Training Candidates (v0.2.1):     200  (100 Curated Dev + 100 OpenHermes)
Validation Candidates (v0.2.1):    20  (20 Curated Dev)
Evaluation-Only:                  200  (100 SecBench + 100 SWE-bench)
Review Queue:                      50  (50 candidate_unknown_license_db)
Excluded:                           0
```

---

## 8. Contamination & Leakage Results

- **Training vs. Evaluation Exact Duplicates**: 0
- **Training vs. Evaluation Normalized Duplicates**: 0
- **SecBench in Training Candidates**: 0
- **SWE-bench in Training Candidates**: 0

---

## 9. Provenance & Traceability Results

- **Training Candidates with Complete Provenance**: 200 / 200 (100%)
- **Validation Candidates with Complete Provenance**: 20 / 20 (100%)
- **Missing Source Details in Training**: 0

---

## 10. Security Data Classification

Breakdown of the 220 active training/validation candidate items:
- `code_generation`: 100
- `vulnerability_remediation`: 20
- `security_review`: 20
- `debugging`: 20
- `agent_trajectories`: 20
- `authorized_lab`: 20
- `authorization_status`: `defensive` / `authorized` / `isolated_lab` (100%)

---

## 11. Azure Status & Resource Verification

- **Azure CPU/GPU resources created**: 0
- **Azure ML jobs submitted**: 0
- **Azure credit spent**: $0

---

## 12. Final Readiness

**Status**: **READY FOR NEXT DATASET EXPANSION**
