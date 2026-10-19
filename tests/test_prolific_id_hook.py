"""The pre-commit hook in scripts/git-hooks refuses a commit that adds anything
that looks like a Prolific id (24 lowercase hex characters standing alone).

Each test commits into a scratch repository whose ``core.hooksPath`` points at
this repository's hooks. The fake ids are built at run time so that this file
itself contains no 24-hex token (the hook would refuse to commit it).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

HOOKS_DIR = REPO_ROOT / "scripts" / "git-hooks"
FAKE_ID = "ab12" * 6  # 24 hex characters, like a Prolific worker/study id
GIT_SHA = "0123456789" * 4  # a 40-hex commit sha contains 24-hex runs


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "core.hooksPath", str(HOOKS_DIR))
    (tmp_path / "README").write_text("start\n")
    _git(tmp_path, "add", "README")
    assert _git(tmp_path, "commit", "-q", "-m", "init").returncode == 0
    return tmp_path


def _commit(repo: Path, name: str, text: str) -> subprocess.CompletedProcess:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", "--", name)
    return _git(repo, "commit", "-q", "-m", f"add {name}")


def test_refuses_an_added_line_with_a_prolific_id(repo: Path) -> None:
    result = _commit(repo, "data/times.csv", f"pid,secs\n{FAKE_ID},197\n")
    assert result.returncode != 0
    assert "data/times.csv:2" in result.stderr
    assert FAKE_ID not in result.stderr  # the refusal does not echo the id


@pytest.mark.parametrize(
    "text",
    [
        f"PROLIFIC_PID={FAKE_ID}&STUDY_ID=x\n",
        f'{{"participant_id_str": "{FAKE_ID}"}}\n',
        f"raw_{FAKE_ID}.csv\n",
    ],
)
def test_refuses_ids_in_urls_json_and_names(repo: Path, text: str) -> None:
    assert _commit(repo, "notes.txt", text).returncode != 0


def test_refuses_an_id_in_a_file_name(repo: Path) -> None:
    result = _commit(repo, f"raw/{FAKE_ID}.csv", "x\n")
    assert result.returncode != 0
    assert "file name" in result.stderr


def test_allows_longer_hex_tokens_and_clean_files(repo: Path) -> None:
    text = f"commit {GIT_SHA}\nsha256 {GIT_SHA * 2}\nplain text\n"
    assert _commit(repo, "ok.txt", text).returncode == 0


def test_allows_removing_an_id(repo: Path) -> None:
    (repo / "old.csv").write_text(f"{FAKE_ID}\n")
    _git(repo, "add", "old.csv")
    assert _git(repo, "commit", "-q", "--no-verify", "-m", "legacy").returncode == 0
    assert _commit(repo, "old.csv", "REDACTED\n").returncode == 0


def test_refuses_a_merge_that_brings_in_an_id(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.csv").write_text(f"{FAKE_ID}\n")
    _git(repo, "add", "side.csv")
    assert _git(repo, "commit", "-q", "--no-verify", "-m", "side").returncode == 0
    _git(repo, "checkout", "-q", "main")
    (repo / "main.txt").write_text("diverge\n")
    _git(repo, "add", "main.txt")
    assert _git(repo, "commit", "-q", "-m", "main").returncode == 0
    result = _git(repo, "merge", "--no-edit", "side")
    assert result.returncode != 0
    assert "side.csv:1" in result.stderr
