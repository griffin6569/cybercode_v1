# CyberCodeMini Data Architecture & Governance (Phase 10.1)

This directory houses the CyberCodeMini dataset tiers, evaluation benchmarks, review queues, and dataset metadata records.

## Data Classification Tiers

CyberCodeMini enforces strict physical separation across data tiers:

1. **`processed/training_candidates/`**: Cleaned, verified, non-benchmark data explicitly cleared for model training.
2. **`processed/validation_candidates/`**: Non-benchmark data reserved for in-domain validation during training runs.
3. **`evaluation/`**: Benchmark evaluation datasets (`secbench`, `swe_bench`) strictly excluded from ordinary training data to prevent benchmark contamination.
4. **`review/`**: Unverified datasets or items with unknown licenses or unclear provenance (`unknown_license`). Held in isolation until audit clearance.
5. **`excluded/`**: Data explicitly rejected for quality, security, authorization, or licensing violations.

> [!IMPORTANT]
> **Data Governance Policy Notice**
> A permissive dataset license (such as MIT or Apache-2.0) does not automatically mean that every underlying artifact (source code file, repository, patch, or third-party content) is cleared for model training. CyberCodeMini treats licensing, provenance, benchmark status, authorization, and data quality as separate approval dimensions.
> *This is an engineering and data-governance policy, not a legal opinion.*
