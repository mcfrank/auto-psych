"""The impossible-model sweep is the control for the literature sweep, so its
config may differ from ``holdout_recovery_faithful.yaml`` only in its ground
truths (``gt_models``) and where they live (``gt_models_dir``).

It had drifted: 2 rounds x 3 candidates instead of 5 x 6, no ``design`` block
(a 32-stimulus design instead of 64), a 900 s agent timeout instead of
1800 s, no ``n_critique_proposals`` and no ``fit.target_accept`` (PyMC's
default instead of 0.8).
"""

from __future__ import annotations

import yaml

from tests.paths import REPO_ROOT

CONFIGS = REPO_ROOT / "scripts" / "subjective_randomness" / "configs"
ALLOWED = {"gt_models", "gt_models_dir"}


def _load(name):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_the_impossible_config_differs_from_the_faithful_one_only_in_its_ground_truths():
    faithful = _load("holdout_recovery_faithful.yaml")
    impossible = _load("impossible_holdout_recovery.yaml")
    assert {k: v for k, v in impossible.items() if k not in ALLOWED} == {
        k: v for k, v in faithful.items() if k not in ALLOWED
    }
    assert set(impossible) - set(faithful) == {"gt_models_dir"}
    assert set(impossible["gt_models"]).isdisjoint(faithful["gt_models"])


SLURM = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"
RESOURCE_OPTIONS = ("--partition", "--time", "--cpus-per-task", "--mem")


def _resources(sbatch_name):
    """The array's own #SBATCH resource requests, option -> value."""
    requests = {}
    for line in (SLURM / sbatch_name).read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH ") and "=" in line:
            option, value = line.removeprefix("#SBATCH ").split("=", 1)
            if option in RESOURCE_OPTIONS:
                requests[option] = value.strip()
    return requests


def test_the_impossible_array_requests_the_literature_arrays_resources():
    """With half the CPUs the impossible cells ran two fits at a time instead
    of four, and 32 GB killed a cell whose harness grew to the ~33 GB the
    literature cells also reach (2026-09-29)."""
    literature = _resources("holdout_recovery_array.sbatch")
    assert set(literature) == set(RESOURCE_OPTIONS)
    assert _resources("impossible_holdout_recovery_array.sbatch") == literature
