"""What makes a checkout's staged code "dirty" (scripts/subjective_randomness/slurm/code_commit.sh)."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subjective_randomness" / "slurm" / "code_commit.sh"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _code(repo) -> str:
    return subprocess.run(["bash", str(SCRIPT), str(repo)], check=True, capture_output=True, text=True).stdout.strip()


def test_root_clutter_is_not_dirty_but_code_and_edits_are(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c")
    clean = _code(repo)
    assert "dirty" not in clean
    (repo / "job_123.out").write_text("log\n")
    (repo / ".secrets~").write_text("backup\n")
    assert _code(repo) == clean
    (repo / "src" / "new.py").write_text("y = 2\n")
    assert "dirty" in _code(repo)
    (repo / "src" / "new.py").unlink()
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert "dirty" in _code(repo)
