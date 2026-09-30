"""A full run's Hosting sites carry its ``run_label``.

Each parallel run deploys to its own site. They were named
``<firebase project>-run<i>``, so every series of runs in one Firebase project
reused (and replaced the pages of) the earlier series' sites. They are now
``<firebase project>-<run_label>-run<i>`` (``auto-psych-2c5da-sr-oct26-run1``).
A name Firebase would refuse (at most 30 characters; lowercase letters, digits
and ``-``) stops the launch before anything is copied or submitted, instead of
failing at the deploy after the page has been built.

``submit_parallel.sh`` and ``start_full_run.sh`` run here from a copy of their
directory with stand-ins for ``_env.sh``, the config bridge and ``sbatch``:
nothing is validated against Prolific, rendered or submitted.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.paths import REPO_ROOT

LIVE = REPO_ROOT / "scripts" / "outer_loop_live"


def _hosting_site(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'source "{LIVE / "_hosting_site.sh"}"; hosting_site "$@"', "_", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_the_site_names_the_project_the_run_label_and_the_run():
    result = _hosting_site("auto-psych-2c5da", "sr-oct26", "run1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "auto-psych-2c5da-sr-oct26-run1"


def test_without_a_run_label_the_site_is_the_project_and_the_run():
    result = _hosting_site("auto-psych-2c5da", "", "run2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "auto-psych-2c5da-run2"


def test_a_site_over_30_characters_is_refused():
    result = _hosting_site("auto-psych-2c5da", "sr-october-2026", "run1")
    assert result.returncode != 0
    assert "auto-psych-2c5da-sr-october-2026-run1" in result.stderr
    assert "30" in result.stderr


def test_a_run_label_with_characters_firebase_refuses_is_refused():
    result = _hosting_site("auto-psych-2c5da", "sr_oct26", "run1")
    assert result.returncode != 0
    assert "sr_oct26" in result.stderr


# --- submit_parallel.sh -----------------------------------------------------

_SUBMIT_ENV = """REPO="$STAND_IN_REPO"
WORK_ROOT="$STAND_IN_WORK_ROOT"
VENV_PY="$(dirname "${BASH_SOURCE[0]}")/stand_in_python"
"""
_STAND_IN_PYTHON = """#!/bin/bash
echo "python $*" >> "$CALLS"
"""
_STAND_IN_SBATCH = """#!/bin/bash
echo "sbatch $*" >> "$CALLS"
echo 12345
"""


@pytest.fixture
def submit(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("submit_parallel.sh", "_hosting_site.sh", "run_live.sbatch"):
        shutil.copy(LIVE / name, scripts / name)
    (scripts / "_env.sh").write_text(_SUBMIT_ENV, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for path, text in (
        (scripts / "stand_in_python", _STAND_IN_PYTHON),
        (bin_dir / "sbatch", _STAND_IN_SBATCH),
    ):
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("stand-in checkout\n", encoding="utf-8")
    calls = tmp_path / "calls.log"

    def run(**extra_env: str):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "STAND_IN_REPO": str(repo),
            "STAND_IN_WORK_ROOT": str(tmp_path / "work"),
            "CALLS": str(calls),
            "K": "2",
            "FIREBASE_PROJECT": "auto-psych-2c5da",
            **extra_env,
        }
        result = subprocess.run(
            ["bash", str(scripts / "submit_parallel.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logged = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        return result, [c for c in logged if c.startswith("sbatch ")]

    return run


def test_each_run_is_submitted_to_a_site_named_after_the_run_label(submit):
    result, submitted = submit(SERIES_LABEL="sr-oct26")

    assert result.returncode == 0, result.stderr
    assert len(submitted) == 2
    assert "AUTO_PSYCH_HOSTING_SITE=auto-psych-2c5da-sr-oct26-run1" in submitted[0]
    assert "AUTO_PSYCH_HOSTING_SITE=auto-psych-2c5da-sr-oct26-run2" in submitted[1]
    assert "RUN_LABEL=run1" in submitted[0]  # the run's own label is unchanged


def test_a_site_name_firebase_would_refuse_submits_nothing(submit):
    result, submitted = submit(SERIES_LABEL="sr-october-2026")

    assert result.returncode != 0
    assert submitted == []


# --- start_full_run.sh ------------------------------------------------------

_START_ENV = """VENV_PY="$(dirname "${BASH_SOURCE[0]}")/stand_in_python"
"""
_START_BRIDGE = """#!/bin/bash
echo "$*" >> "$CALLS"
if [[ " $* " == *" --check "* ]]; then
  echo "export PROJECT=subjective_randomness N_EXPERIMENTS=3 N_PARTICIPANTS=40"
  echo "export PROLIFIC_MODE=live WALLTIME=1-23:00:00 QOS="
  echo "export RUN_LABEL=$STAND_IN_RUN_LABEL FIREBASE_PROJECT=auto-psych-2c5da"
fi
"""
_START_SUBMIT = """#!/bin/bash
echo "submit_parallel RUNS=$RUNS SERIES_LABEL=$SERIES_LABEL" >> "$CALLS"
"""


@pytest.fixture
def start(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("start_full_run.sh", "_hosting_site.sh"):
        shutil.copy(LIVE / name, scripts / name)
    (scripts / "_env.sh").write_text(_START_ENV, encoding="utf-8")
    (scripts / "stand_in_python").write_text(_START_BRIDGE, encoding="utf-8")
    (scripts / "stand_in_python").chmod(0o755)
    (scripts / "submit_parallel.sh").write_text(_START_SUBMIT, encoding="utf-8")
    (scripts / "full_run.yaml").write_text("project: subjective_randomness\n", encoding="utf-8")
    calls = tmp_path / "calls.log"

    def run(run_label: str, answer: str):
        result = subprocess.run(
            ["bash", str(scripts / "start_full_run.sh")],
            input=answer + "\n",
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "WORK_ROOT": str(tmp_path / "work"),
                "RUNS": "1",
                "CALLS": str(calls),
                "STAND_IN_RUN_LABEL": run_label,
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        logged = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        return result, logged

    return run


def test_the_launch_summary_names_the_sites_and_passes_the_run_label_on(start):
    result, calls = start("sr-oct26", "yes")

    assert result.returncode == 0, result.stderr
    assert "auto-psych-2c5da-sr-oct26-run1" in result.stdout
    assert calls[-1] == "submit_parallel RUNS=1 SERIES_LABEL=sr-oct26"


def test_a_run_label_too_long_for_a_site_stops_before_the_prompt(start):
    result, calls = start("sr-october-2026", "yes")

    assert result.returncode != 0
    assert "auto-psych-2c5da-sr-october-2026-run1" in result.stderr
    assert "Type" not in result.stderr  # never reached the typed-"yes" prompt
    assert not any(c.startswith("submit_parallel") for c in calls)
    assert not any("--render-only" in c for c in calls)
