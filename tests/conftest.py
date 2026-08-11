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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: slow tests (NUTS sampling); skipped by default with -m 'not slow'",
    )


@pytest.fixture
def fixtures_dir():
    """Path to tests/fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def project_id():
    return "subjective_randomness"


@pytest.fixture
def run_id():
    return 1
