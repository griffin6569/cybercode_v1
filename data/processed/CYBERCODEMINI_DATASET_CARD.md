# CyberCodeMini Dataset Card (Corpus v0.3.0)

## 1. Dataset Overview
- **Dataset Name**: `CyberCodeMini Corpus`
- **Corpus Version**: `v0.3.0`
- **Primary Model Target**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Domain**: Cybersecurity Software Engineering, Secure Coding, Vulnerability Remediation, Debugging, and Agentic Tool Use.
- **Total Training Candidates**: 900
- **Total Validation Candidates**: 90
- **Total Evaluation Benchmark Items**: 200 (`secbench`: 100, `swe_bench`: 100)
- **Review Queue Items**: 50 (`candidate_unknown_license_db`)

---

## 2. Capabilities Taxonomy & Target Mix
The dataset is structured across 11 core capabilities to balance small-model performance:
1. **Vulnerability Remediation** (19.8%): Parameterized fixes for SQLi, XSS, Command Injection, Path Traversal, SSRF, Deserialization, and Crypto flaws.
2. **Code Generation** (19.3%): Python, C/C++ algorithms, thread-safe data structures, API handlers, and system code.
3. **Agent / Tool Use** (14.4%): Multi-turn tool execution trajectories (list_files, read_file, edit_file, run_tests, diagnose).
4. **Security Review** (10.4%): Code auditing, risk evaluation, and CWE mapping.
5. **Authorized Security Labs & CTF** (8.4%): Isolated sandbox lab tasks and authorized CTF remediation scenarios.
6. **Vulnerability Detection** (8.3%): Detection and classification of security bugs.
7. **Debugging** (6.9%): Traceback diagnosis, exception handling, and code repair.
8. **Refactoring** (5.2%): Code modernization and quality enhancement.
9. **Test Generation** (5.2%): pytest/unittest unit test creation.
10. **Agent Trajectories** (1.9%): Trajectory planning and file manipulation.

---

## 3. Data Governance & Licensing Policy
CyberCodeMini enforces strict multi-dimensional data governance:
- **Dataset License vs. Underlying Content**: Permissive dataset-level licenses (MIT / Apache-2.0 / Llama-2) do not automatically clear third-party repository artifacts. Licensing, provenance, benchmark status, authorization context, and quality are evaluated as separate approval dimensions.
- **Benchmark Separation Policy**: Benchmark evaluation datasets (`SecBench`, `SWE-bench`) are strictly classified as `evaluation_only` and prohibited from entering training candidates.
- **Security Authorization Context**: All security examples require explicit authorization metadata (`defensive`, `authorized`, `isolated_lab`).
- **Unknown License Isolation**: Unverified datasets (`candidate_unknown_license_db`) are held in `data/review/` and excluded from training.

---

## 4. Train / Validation / Evaluation Tiers
- **`processed/training_candidates/`**: 900 verified non-benchmark training items (`v0.3.0`).
- **`processed/validation_candidates/`**: 90 held-out validation items (`v0.3.0`).
- **`evaluation/`**: 200 evaluation benchmark items (SecBench + SWE-bench).
- **`review/`**: 50 unverified items under review.

---

## 5. Quality & Security Assurance
- **Deduplication**: 0 exact or normalized duplicates between training candidates and evaluation benchmarks.
- **PII & Secret Scanning**: 0 unreviewed high-entropy keys or credentials in training.
- **Monoculture Control**: Maximum single source contribution is 17.8% (well under the 40% warning threshold).
