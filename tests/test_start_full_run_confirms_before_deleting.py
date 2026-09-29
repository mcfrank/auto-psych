"""``start_full_run.sh`` deletes nothing until the user has typed "yes".

It cleared the selected runs' output and run-copy directories
(``$WORK_ROOT/run<i>``, ``$WORK_ROOT/runs/run<i>``) before its typed-"yes"
prompt, so answering "no" still destroyed the earlier runs' results. The
launcher is run here from a copy of its directory whose ``_env.sh``, config
bridge and ``submit_parallel.sh`` are stand-ins: nothing is validated against
Prolific, rendered or submitted.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.paths import REPO_ROOT

LAUNCHER = REPO_ROOT / "scripts" / "outer_loop_live" / "start_full_run.sh"

_STAND_IN_ENV = """VENV_PY="$(dirname "${BASH_SOURCE[0]}")/stand_in_python"
"""
# The config bridge: --check prints the settings the launcher evals.
_STAND_IN_PYTHON = """#!/bin/bash
echo "$*" >> "$CALLS"
if [[ " $* " == *" --check "* ]]; then
  echo "export PROJECT=subjective_randomness N_EXPERIMENTS=3 N_PARTICIPANTS=40"
  echo "export PROLIFIC_MODE=live WALLTIME=1-23:00:00 QOS="
fi
"""
_STAND_IN_SUBMIT = """#!/bin/bash
echo "submit_parallel $RUNS" >> "$CALLS"
"""


@pytest.fixture
def launch(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(LAUNCHER, scripts / "start_full_run.sh")
    (scripts / "_env.sh").write_text(_STAND_IN_ENV, encoding="utf-8")
    (scripts / "stand_in_python").write_text(_STAND_IN_PYTHON, encoding="utf-8")
    (scripts / "stand_in_python").chmod(0o755)
    (scripts / "submit_parallel.sh").write_text(_STAND_IN_SUBMIT, encoding="utf-8")
    (scripts / "full_run.yaml").write_text(
        "project: subjective_randomness\n", encoding="utf-8"
    )
    work_root = tmp_path / "work"
    earlier = [work_root / "run1", work_root / "runs" / "run1", work_root / "run2"]
    for directory in earlier:
        directory.mkdir(parents=True)
        (directory / "results.csv").write_text("earlier results\n", encoding="utf-8")
    calls = tmp_path / "calls.log"

    def run(answer: str):
        result = subprocess.run(
            ["bash", str(scripts / "start_full_run.sh")],
            input=answer + "\n",
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "WORK_ROOT": str(work_root),
                "K": "2",
                "CALLS": str(calls),
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        logged = (
            calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
        )
        return result, logged

    return run, earlier


def test_answering_no_deletes_nothing_and_launches_nothing(launch):
    run, earlier = launch

    result, calls = run("no")

    assert result.returncode != 0
    assert all((d / "results.csv").exists() for d in earlier)
    assert not any(c.startswith("submit_parallel") for c in calls)
    assert not any("--render-only" in c for c in calls)


def test_the_prompt_lists_exactly_what_will_be_deleted(launch):
    run, earlier = launch

    result, _ = run("no")

    for directory in earlier:
        assert str(directory) in result.stdout
    assert "runs/run2" not in result.stdout  # does not exist, so is not listed


def test_answering_yes_deletes_the_listed_dirs_then_launches(launch):
    run, earlier = launch

    result, calls = run("yes")

    assert result.returncode == 0, result.stderr
    assert not any(d.exists() for d in earlier)
    assert calls[-1] == "submit_parallel "
