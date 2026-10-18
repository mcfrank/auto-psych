"""The job environments build their venvs on uv-managed Python only.

On coco, miniconda's bin (for uv itself) is on PATH, and the node has a
system Python 3.12 too; uv's default preference merely *prefers* its managed
interpreters, so a missing managed one would silently build the venv on
conda's or the system's Python. ``only-managed`` makes that a loud failure.
"""

from __future__ import annotations

import pytest

from tests.paths import REPO_ROOT


@pytest.mark.parametrize("script", [
    REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "_env.sh",  # sweeps, and RSA (sources it)
    REPO_ROOT / "scripts" / "outer_loop_live" / "_env.sh",
])
def test_the_job_environment_allows_only_uv_managed_python(script):
    assert 'export UV_PYTHON_PREFERENCE=only-managed' in script.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [
    REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "_env.sh",
    REPO_ROOT / "scripts" / "outer_loop_live" / "_env.sh",
])
def test_the_job_environment_drops_an_inherited_activated_environment(script):
    # sbatch --export=ALL hands a job the submitting shell's activated conda
    # env or virtualenv, which uv takes as the environment to use.
    assert "unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV" in script.read_text(encoding="utf-8")
