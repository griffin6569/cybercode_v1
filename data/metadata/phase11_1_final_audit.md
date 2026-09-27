# CyberCodeMini Phase 11.1 — Final Corpus Forensic Audit Report

**Date**: 2026-09-27  
**Phase**: Phase 11.1  
**Corpus Version**: v0.3.0  
**Status**: READY_FOR_FREEZE  

---

## 1. Executive Summary

Phase 11.1 conducted a complete forensic row-level audit of the authoritative CyberCodeMini Corpus `v0.3.0` (`cybercodemini_train_candidates_v0.3.0.jsonl` and `cybercodemini_validation_candidates_v0.3.0.jsonl`). 

All 900 training candidates and 90 validation candidates were evaluated across schema validity, structural agent trajectory tiers, capability coverage, source/license distribution, quality scores, and benchmark contamination.

**Key Forensic Findings**:
- **Row Counts Verified**: 900 training candidates, 90 validation candidates, 200 evaluation benchmark items (`SecBench`: 100, `SWE-bench`: 100), and 50 review queue items (`candidate_unknown_license_db`).
- **Benchmark Protection**: 0 benchmark contamination instances across SecBench and SWE-bench.
- **Agent Trajectories**: 147 training examples (16.3%) and 16 validation examples (17.8%) qualify as full multi-step agent trajectories with complete plan $\rightarrow$ search $\rightarrow$ edit $\rightarrow$ test $\rightarrow$ diagnose $\rightarrow$ fix $\rightarrow$ pass execution loops.
- **Quality Score**: Mean quality score of 97.78/100, with 0 items below the project threshold (70.0).
- **Schema & Provenance**: 100% schema validity (990/990 rows) and 100% traceable provenance.

---

## 2. Corpus Accounting & Tier Separation

| Tier | File Path | Actual Verified Rows | Classification Status |
| :--- | :--- | :--- | :--- |
| **Training Candidates** | `data/processed/training_candidates/cybercodemini_train_candidates_v0.3.0.jsonl` | 900 | `training_candidate` |
| **Validation Candidates** | `data/processed/validation_candidates/cybercodemini_validation_candidates_v0.3.0.jsonl` | 90 | `validation_candidate` |
| **SecBench Evaluation** | `data/evaluation/secbench/examples.jsonl` | 100 | `evaluation_only` |
| **SWE-bench Evaluation** | `data/evaluation/swe_bench/examples.jsonl` | 100 | `evaluation_only` |
| **Review Queue** | `data/review/unknown_license/examples.jsonl` | 50 | `review` |
| **Total Inspected** | | **1,240** | |

---

## 3. Capability Taxonomy Distribution (Training Candidates)

| Canonical Primary Capability | Count | Percentage | Secondary Capabilities Carried |
| :--- | :--- | :--- | :--- |
| `vulnerability_remediation` | 178 | 19.8% | `secure_coding`, `security_reasoning` |
| `code_generation` | 174 | 19.3% | `software_engineering` |
| `agent_tool_use` | 130 | 14.4% | `agent_trajectories`, `repository_navigation`, `software_engineering` |
| `security_review` | 94 | 10.4% | `security_reasoning`, `vulnerability_detection` |
| `authorized_lab` | 76 | 8.4% | `ctf`, `security_reasoning` |
| `vulnerability_detection` | 75 | 8.3% | `security_reasoning`, `security_review` |
| `debugging` | 62 | 6.9% | `software_engineering` |
| `refactoring` | 47 | 5.2% | `software_engineering` |
| `test_generation` | 47 | 5.2% | `software_engineering` |
| `agent_trajectories` | 17 | 1.9% | `agent_tool_use`, `repository_navigation` |
| **Total** | **900** | **100.0%** | |

---

## 4. Agent Trajectory Structural Tier Audit

| Structural Agent Tier | Criteria | Training Count | Validation Count | % of Corpus |
| :--- | :--- | :--- | :--- | :--- |
| `simple_tool_use` | Text mention of tools (0 tool execution blocks) | 0 | 0 | 0.0% |
| `single_step_tool_use` | 1 tool call turn | 0 | 0 | 0.0% |
| `multi_step_tool_use` | 2–3 tool call turns | 0 | 0 | 0.0% |
| `full_agent_trajectory` | 4+ tool call turns (search $\rightarrow$ read $\rightarrow$ edit $\rightarrow$ test) | 147 | 16 | 16.3% |
| **Total Agent Items** | | **147** | **16** | |

---

## 5. Vulnerability Remediation Deep Dive Audit

- **Remediation Examples Inspected**: 178 / 178
- **Examples with Actual Patch**: 178 (100.0%)
- **Examples with Security Explanation**: 178 (100.0%)
- **Examples with Verification/Test Context**: 178 (100.0%)
- **Vulnerability Classes Covered**: SQL Injection (CWE-89), XSS (CWE-79), Command Injection (CWE-78), Path Traversal (CWE-22), SSRF (CWE-918), Insecure Deserialization (CWE-502), Hardcoded Secrets (CWE-798), Broken Authentication (CWE-287), Unsafe Upload (CWE-434), Race Conditions (CWE-367).

---

## 6. Source, License & Security Authorization Audit

### Source Distribution
- `cybercode_curated_vulnerability_remediation`: 160 (17.8%)
- `cybercode_curated_security_review_and_reasoning`: 150 (16.7%)
- `cybercode_curated_debugging_and_refactoring`: 140 (15.6%)
- `cybercode_curated_agent_trajectories`: 130 (14.4%)
- `cybercode_curated_dev`: 100 (11.1%)
- `OpenHermes-Code-Sample`: 100 (11.1%)
- `cybercode_curated_authorized_labs`: 60 (6.7%)
- `CodeLlama-Instruct-Code-Samples`: 60 (6.7%)

### License Breakdown
- `MIT`: 740 (82.2%)
- `Apache-2.0`: 100 (11.1%)
- `Llama-2`: 60 (6.7%)
- `License Status`: 100% Verified

### Security Authorization Distribution
- `defensive`: 477 (53.0%)
- `authorized`: 207 (23.0%)
- `not_applicable`: 200 (22.2%)
- `explicit`: 16 (1.8%)

---

## 7. Quality Score & Integrity Analysis

- **Mean Quality Score**: 97.78 / 100
- **Median Quality Score**: 100.00 / 100
- **P25 / P75 / P95**: 95.0 / 100.0 / 100.0
- **Items Below Project Threshold (70.0)**: 0
- **Training vs. Validation Duplicates**: 0
- **Training vs. Evaluation Duplicates**: 0
- **Validation vs. Evaluation Duplicates**: 0
- **Schema Errors**: 0 (990 / 990 valid Pydantic models)

---

## 8. Findings & Categorization

> [!NOTE]
> **PASS Findings**:
> - All 900 training candidates and 90 validation candidates pass row-level schema validation.
> - Evaluation benchmarks (`SecBench` and `SWE-bench`) are 100% isolated.
> - No monoculture risk detected (max single-source share is 17.8%).
> - All capability gap reporting scripts default to `v0.3.0`.

> [!TIP]
> **WARNING Findings**: None.

> [!CAUTION]
> **REVIEW REQUIRED Findings**: None.

---

## 9. Azure Status & Resource Verification

- **Azure resources created**: 0
- **Azure training jobs**: 0
- **Azure credit spent**: $0

---

## 10. Final Status Decision

**Status**: **READY_FOR_FREEZE**
