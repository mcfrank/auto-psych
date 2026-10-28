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
