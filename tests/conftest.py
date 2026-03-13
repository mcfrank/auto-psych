"""Shared pytest fixtures for auto-psych tests."""

import tempfile
from pathlib import Path

import pytest

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def temp_run_dir(tmp_path):
    """A temporary directory that looks like a run dir (with agent subdirs and fixture copies)."""
    for key in ["1theorist", "2experiment_designer", "3experiment_implementer", "4deployer", "5simulated_participant", "6data_analyst", "7interpreter"]:
        (tmp_path / key).mkdir(exist_ok=True)
    # Copy key fixtures so validators can find them
    (tmp_path / "1theorist" / "models_manifest.yaml").write_text((FIXTURES_DIR / "models_manifest.yaml").read_text())
    (tmp_path / "2experiment_designer" / "stimuli.json").write_text((FIXTURES_DIR / "stimuli.json").read_text())
    (tmp_path / "3experiment_implementer" / "index.html").write_text((FIXTURES_DIR / "3experiment_implementer" / "index.html").read_text())
    (tmp_path / "4deployer" / "config.json").write_text((FIXTURES_DIR / "4deployer" / "config.json").read_text())
    return tmp_path


@pytest.fixture
def project_id():
    return "subjective_randomness"


@pytest.fixture
def run_id():
    return 1
