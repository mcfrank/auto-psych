"""Shared pytest fixtures for auto-psych tests."""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path so "import src..." works when running pytest
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def temp_run_dir(tmp_path):
    """A temporary directory that looks like a run dir (with agent subdirs and fixture copies)."""
    for key in ["1_theory", "2_design", "3_implement", "4_collect", "5_analyze", "6_interpret"]:
        (tmp_path / key).mkdir(exist_ok=True)
    # Copy key fixtures so validators can find them
    (tmp_path / "1_theory" / "models_manifest.yaml").write_text((FIXTURES_DIR / "models_manifest.yaml").read_text())
    (tmp_path / "2_design" / "stimuli.json").write_text((FIXTURES_DIR / "stimuli.json").read_text())
    (tmp_path / "3_implement" / "index.html").write_text((FIXTURES_DIR / "3_implement" / "index.html").read_text())
    (tmp_path / "3_implement" / "config.json").write_text((FIXTURES_DIR / "3_implement" / "config.json").read_text())
    return tmp_path


@pytest.fixture
def project_id():
    return "subjective_randomness"


@pytest.fixture
def run_id():
    return 1
