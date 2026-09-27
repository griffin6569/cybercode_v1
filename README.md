# CyberCodeMini

**CyberCodeMini** is a specialized open-weight coding + cybersecurity + agentic AI model targeting software engineering and authorized security tasks (vulnerability analysis, secure code review, remediation, CTF labs, and sandboxed agent workflows).

---

## Architecture & Data Pipeline

### 1. Canonical Dataset Schema
All training examples follow the canonical Pydantic v2 schema in [`datasets/schemas/schema.py`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/datasets/schemas/schema.py):
- **Messages**: Multi-turn support (`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`, `TOOL_RESULT`).
- **Tool Calls**: Structured schema (`name`, `arguments`, `tool_call_id`).
- **Security Context Metadata**: Enforces explicit `authorization` (`authorized`, `explicit`, `defensive`, `educational`, `unknown`), `environment` (`isolated_lab`, `educational`, `ctf`), `cve_ids`, `cwe_ids`, and `source_id`.

### 2. Chat Formatting
Implemented in [`datasets/formatting/chat_formatter.py`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/datasets/formatting/chat_formatter.py):
- Loads official model tokenizer templates (`Qwen/Qwen2.5-Coder-1.5B-Instruct`).
- Fallback ChatML formatter (`<|im_start|>role\ncontent<|im_end|>`) preserving tool calls and tool outputs deterministically.

### 3. Tokenization & Loss Masking
Implemented in [`datasets/tokenization.py`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/datasets/tokenization.py):
- **Why Loss Masking is Used**: To prevent gradient computation on system prompts, user queries, and environmental observations (tool results). Learning is focused on model-generated assistant responses and tool call actions.
- **Configurable Strategies** (set in `configs/training.yaml`):
  - `assistant_only`: Masks system, user, tool, and tool_result tokens (`labels = -100`).
  - `assistant_and_tool_outputs` (Default): Masks system, user, and tool_result tokens (`labels = -100`), trains on assistant responses and tool calls.
  - `all_messages`: Trains on all message tokens.

### 4. Truncation & Caching
- **Max Sequence Length**: Configured via `configs/model.yaml` (default `2048`).
- **Truncation Tracking**: Tracks `was_truncated`, `original_token_count`, and `final_token_count`.
- **Deterministic Cache**: Caches tokenized outputs in `data/interim/` keyed by dataset hash, tokenizer revision, max sequence length, and loss masking strategy.

### 5. Data Collator
Implemented in [`training/collator.py`](file:///c:/Users/user/OneDrive/Documents/Desktop/cybercode/training/collator.py):
- `CyberCodeDataCollator` handles dynamic batch padding (`input_ids`, `attention_mask`, `labels` padded with `-100`).
- Configurable sequence packing (`packing: false` default).

---

## Tokenization Smoke Test & Inspection

Run the Phase 8 tokenization smoke test (CPU-compatible):
```bash
python scripts/tokenization_smoke_test.py
```

Inspect loss masking for any dataset example:
```bash
python scripts/inspect_loss_mask.py -idx 0 -s assistant_and_tool_outputs
```

Run test suite:
```bash
pytest tests/ -v
```
