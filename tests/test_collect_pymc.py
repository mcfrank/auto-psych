"""Tests for PyMC-based synthetic data generation in collect.

`_generate_from_pymc_models` samples synthetic responses from the theorist's
PyMC models' prior-predictive p_left (no MCMC fit — the prior is the generative
distribution for synthetic participants). Each raw stimulus is featurized first
so the models can read their `pm.Data` columns.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipelines.outer_loop.collect import (
    _generate_from_models,
    _generate_from_pymc_models,
)
from tests.paths import PYMC_MODEL_FIXTURES_DIR

FEATURIZE = (
    Path(__file__).resolve().parent.parent
    / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _seed(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(PYMC_MODEL_FIXTURES_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml"
    )
    return models_dir


@pytest.mark.slow
def test_generate_from_pymc_models_shapes_and_columns(tmp_path):
    models_dir = _seed(tmp_path)
    stimuli = [
        {"sequence_a": "HHHT", "sequence_b": "HTHT"},
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"},
    ]
    rows = _generate_from_pymc_models(
        stimuli,
        ["bayesian_fair_coin", "representativeness"],
        n_participants=3,
        models_dir=models_dir,
        featurize_path=FEATURIZE,
        n_samples=100,
        seed=0,
    )

    assert len(rows) == 3 * len(stimuli)
    for r in rows:
        assert r["chose_left"] in (0, 1)
        assert r["chose_right"] == 1 - r["chose_left"]
        assert r["sequence_a"] and r["sequence_b"]
        assert r["model"] in {"bayesian_fair_coin", "representativeness"}
        assert {"participant_id", "trial_index"} <= set(r)


def test_generate_from_models_raises_when_the_model_cannot_be_resolved():
    """A model that produces no prediction must fail loudly, not coin-flip data.

    This used to surface as ``RuntimeError("no prediction")``: the loader error
    was swallowed inside ``get_model_predictions``, which returned ``{}``, and
    ``_generate_from_models`` noticed the empty dict one frame later. Since the
    fail-loud sweep the original error propagates instead, so the message names
    the actual problem (no theorist_dir) rather than its symptom.
    """
    stimuli = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    with pytest.raises(KeyError, match="theorist_dir required"):
        _generate_from_models(
            stimuli, ["nonexistent_model"], n_participants=1, theorist_dir=None
        )


@pytest.mark.slow
def test_generate_is_deterministic_under_fixed_seed(tmp_path):
    models_dir = _seed(tmp_path)
    stimuli = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    kw = dict(models_dir=models_dir, featurize_path=FEATURIZE, n_samples=100, seed=7)
    rows_a = _generate_from_pymc_models(stimuli, ["bayesian_fair_coin"], 2, **kw)
    rows_b = _generate_from_pymc_models(stimuli, ["bayesian_fair_coin"], 2, **kw)
    assert [r["chose_left"] for r in rows_a] == [r["chose_left"] for r in rows_b]
    # Counterbalancing is part of the generative process, so it must also be
    # reproducible under a fixed seed.
    assert [r["sequence_a"] for r in rows_a] == [r["sequence_a"] for r in rows_b]


def test_synthetic_generation_counterbalances_sides(tmp_path):
    """Side is randomized per trial: across participants, BOTH sequences of a pair
    appear in the left (sequence_a) slot, and each row still records the original
    pair (just possibly swapped) as the presented left/right order."""
    models_dir = _seed(tmp_path)
    stimuli = [{"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"}]
    rows = _generate_from_pymc_models(
        stimuli,
        ["bayesian_fair_coin"],
        40,
        models_dir=models_dir,
        featurize_path=FEATURIZE,
        n_samples=50,
        seed=0,
    )
    left_seqs = {r["sequence_a"] for r in rows}
    assert left_seqs == {"HHHHHHHH", "HTHTHTHT"}  # both shown on the left
    for r in rows:
        assert {r["sequence_a"], r["sequence_b"]} == {"HHHHHHHH", "HTHTHTHT"}
        assert r["chose_right"] == 1 - r["chose_left"]
