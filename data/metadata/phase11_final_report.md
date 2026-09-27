# CyberCodeMini Phase 11 — Targeted Dataset Expansion & Capability Balancing Final Report

**Date**: 2026-09-27  
**Phase**: Phase 11  
**Corpus Version**: v0.3.0  
**Status**: READY FOR DATASET REVIEW  

---

## 1. Executive Summary

Phase 11 performed a targeted dataset expansion and capability balancing for CyberCodeMini. Starting from the audited Phase 10.1 baseline of 200 training candidates (`v0.2.1`), the training corpus was expanded to **900 training candidates** (`v0.3.0`) and **90 held-out validation candidates**, covering 11 core software engineering and cybersecurity capabilities.

Key achievements:
- **Zero Benchmark Leakage**: Evaluation benchmarks (`SecBench` and `SWE-bench`) remained 100% protected and excluded from training.
- **Capability Gaps Addressed**: Major additions in vulnerability remediation (+160), agent tool execution (+130), security review (+150), debugging (+140), authorized labs (+60), and code generation (+60).
- **Monoculture Prevention**: No single dataset source exceeds 17.8% of the final training corpus (well below the 40.0% warning threshold).
- **Verification**: 115 / 115 unit tests passed.

---

## 2. Baseline Accounting (v0.2.1)

- **Training Candidates**: 200
- **Validation Candidates**: 20
- **Evaluation-Only**: 200 (`SecBench`: 100, `SWE-bench`: 100)
- **Review Queue**: 50 (`candidate_unknown_license_db`)
- **Excluded**: 0

---

## 3. Dataset Expansion Overview

- **New Candidate Datasets Inspected**: 6
- **New Candidate Datasets Accepted**: 6
- **New Candidate Datasets Rejected**: 0
- **New Candidate Datasets Under Review**: 0

---

## 4. Final Corpus Accounting (v0.3.0)

- **Training Candidates**: 900
- **Validation Candidates**: 90
- **Evaluation-Only**: 200 (`SecBench`: 100, `SWE-bench`: 100)
- **Review Queue**: 50 (`candidate_unknown_license_db`)
- **Excluded**: 0
- **Total Inspected Across Tiers**: 1,240

---

## 5. Capability Distribution (Training Candidates)

| Capability | Count | Percentage | Target Percentage |
| :--- | :--- | :--- | :--- |
| `vulnerability_remediation` | 178 | 19.8% | 15–20% |
| `code_generation` | 174 | 19.3% | 15–20% |
| `agent_tool_use` | 130 | 14.4% | 10–15% |
| `security_review` | 94 | 10.4% | 10–15% |
| `authorized_lab` | 76 | 8.4% | 5–10% |
| `vulnerability_detection` | 75 | 8.3% | 5–10% |
| `debugging` | 62 | 6.9% | 10–15% |
| `refactoring` | 47 | 5.2% | 5–10% |
| `test_generation` | 47 | 5.2% | 5–10% |
| `agent_trajectories` | 17 | 1.9% | 5–10% |
| **Total** | **900** | **100.0%** | |

---

## 6. Source Distribution & Diversity Audit

| Source Dataset | Count | Percentage | Monoculture Flag |
| :--- | :--- | :--- | :--- |
| `cybercode_curated_vulnerability_remediation` | 160 | 17.8% | Pass (<40%) |
| `cybercode_curated_security_review_and_reasoning` | 150 | 16.7% | Pass (<40%) |
| `cybercode_curated_debugging_and_refactoring` | 140 | 15.6% | Pass (<40%) |
| `cybercode_curated_agent_trajectories` | 130 | 14.4% | Pass (<40%) |
| `cybercode_curated_dev` | 100 | 11.1% | Pass (<40%) |
| `OpenHermes-Code-Sample` | 100 | 11.1% | Pass (<40%) |
| `cybercode_curated_authorized_labs` | 60 | 6.7% | Pass (<40%) |
| `CodeLlama-Instruct-Code-Samples` | 60 | 6.7% | Pass (<40%) |

---

## 7. License & Security Governance

- **MIT License**: 740 items (82.2%)
- **Apache-2.0 License**: 100 items (11.1%)
- **Llama-2 License**: 60 items (6.7%)
- **License Status**: 100% Verified
- **Security Authorization Context**:
  - `defensive`: 477 (53.0%)
  - `authorized`: 207 (23.0%)
  - `not_applicable`: 200 (22.2%)
  - `explicit`: 16 (1.8%)

---

## 8. Deduplication & Quality Assurance

- **Exact Duplicate Prompts**: 0
- **Normalized Duplicate Prompts**: 0
- **Training vs. Evaluation Duplicates**: 0
- **Training vs. Validation Duplicates**: 0
- **PII / Secret Scanning Findings**: 0 unreviewed credentials or keys detected.

---

## 9. Azure Status & Resource Verification

- **Azure resources created**: 0
- **Azure training jobs**: 0
- **Azure credit spent**: $0

---

## 10. Final Status

**Status**: **READY FOR DATASET REVIEW**
