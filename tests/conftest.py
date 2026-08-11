"""Shared pytest fixtures for auto-psych tests.

Filesystem paths and the standalone-script loader live in ``tests/paths.py``;
they are plain module constants because most test modules need them at import
time (in a decorator, or to build another path), which a fixture cannot serve.
"""

import sys
from pathlib import Path

import pytest

# Bootstrap: put the repo root on sys.path so `import src...` works, before
# anything (including tests.paths) is imported from it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.paths import FIXTURES_DIR  # noqa: E402  (needs the bootstrap above)


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
