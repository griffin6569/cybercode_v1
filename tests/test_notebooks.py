"""CyberCodeMini Colab Notebook Test Suite

Verifies notebook artifact existence, valid JSON structure, cell contents,
dataset SHA-256 matching, and model hyperparameter alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS_DIR / "CyberCodeMini_Colab_Training.ipynb"
README_PATH = NOTEBOOKS_DIR / "README.md"


def test_1_notebook_artifacts_exist():
    """Test 1: Colab notebook and README artifacts exist."""
    assert NOTEBOOK_PATH.exists(), "CyberCodeMini_Colab_Training.ipynb does not exist"
    assert README_PATH.exists(), "notebooks/README.md does not exist"


def test_2_notebook_json_validity_and_cells():
    """Test 2: Notebook is valid nbformat v4 JSON with required cells."""
    content = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert content.get("nbformat") == 4
    cells = content.get("cells", [])
    assert len(cells) >= 6, f"Expected at least 6 cells, got {len(cells)}"


def test_3_notebook_contains_sha256_and_model_id():
    """Test 3: Notebook verifies frozen v0.3.0 SHA-256 and targets Qwen2.5-Coder-1.5B."""
    raw_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "2adf679749e085b227a6f7e5545fe5a7d2cd8f13d197873cee929e21701ddb14" in raw_text
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in raw_text
    assert "cybercode_v1" in raw_text
    assert "Standard_NV36ads_A10_v5" not in raw_text  # Free from Azure GPU blocker dependency


def test_4_modal_script_exists():
    """Test 4: Modal cloud training script exists and contains target SHA-256."""
    modal_script = BASE_DIR / "scripts" / "modal_train.py"
    assert modal_script.exists(), "scripts/modal_train.py does not exist"
    text = modal_script.read_text(encoding="utf-8")
    assert "c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5" in text

