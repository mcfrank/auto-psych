"""Grouped cross-validation as the inner loop's model-selection criterion.

Trial-level PSIS-LOO asks how well a model predicts another answer to a
stimulus it was fitted to; every other answer to that stimulus (and, for a
model with participant parameters, the rest of that participant's session) is
in the fit. Grouped CV holds out whole units instead: a model is fitted on
k - 1 folds and scores the held-out fold's answers by their log posterior
predictive density. ``cv_group`` names the unit: CSV columns, or ``stimulus``
(the task's stimulus identity: the unordered pair for the coin-flip task, so
both presentations of a pair are one unit).
"""

from __future__ import annotations

import csv
import shutil

import numpy as np
import pytest

from src.model_comparison.cv import LOO, Selection, assign_folds, cv_units
from src.model_comparison.posterior import compare_table, model_posterior
from tests.paths import PYMC_MODEL_FIXTURES_DIR

ROWS = [
    {"sequence_a": "HT", "sequence_b": "HH", "participant_id": "0", "trial_index": "0", "chose_left": "1"},
    {"sequence_a": "HH", "sequence_b": "HT", "participant_id": "1", "trial_index": "0", "chose_left": "0"},
    {"sequence_a": "TT", "sequence_b": "HH", "participant_id": "0", "trial_index": "1", "chose_left": "1"},
    {"sequence_a": "HTH", "sequence_b": "THT", "participant_id": "1", "trial_index": "1", "chose_left": "1"},
]


def test_stimulus_units_keep_both_presentations_of_a_pair_together():
    units = cv_units(ROWS, "stimulus")
    assert units[0] == units[1] != units[2]


def test_column_units_group_by_the_named_columns():
    units = cv_units(ROWS, ("participant_id",))
    assert units[0] == units[2] != units[1] == units[3]


def test_folds_hold_whole_units_and_are_deterministic():
    units = [f"u{i // 3}" for i in range(30)]  # 10 units of 3 trials
    folds = assign_folds(units, k=5, seed=1)
    assert sorted(np.bincount(folds)) == [6] * 5
    for unit in set(units):
        assert len({f for u, f in zip(units, folds) if u == unit}) == 1
    assert list(folds) == list(assign_folds(units, k=5, seed=1))
    with pytest.raises(ValueError, match="fewer units"):
        assign_folds(units[:6], k=5)


def test_selection_defaults_to_trial_level_loo():
    assert LOO.criterion == "loo" and Selection() == LOO


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(PYMC_MODEL_FIXTURES_DIR, d)
    return d


@pytest.mark.slow
def test_grouped_cv_compare_table_ranks_by_the_held_out_density(models_dir, tmp_path):
    selection = Selection(criterion="grouped_cv", cv_group="stimulus", cv_folds=3)
    fit = {"draws": 200, "tune": 200, "chains": 2}
    responses = models_dir / "responses.csv"
    table = compare_table(responses, models_dir, cache_dir=tmp_path / "cache", selection=selection, **fit)
    posterior = model_posterior(responses, models_dir, cache_dir=tmp_path / "cache", selection=selection, **fit)

    assert set(table) == {"bayesian_fair_coin", "representativeness"}
    ranked = sorted(table, key=lambda name: table[name]["rank"])
    best, other = ranked
    assert table[best]["elpd_loo"] >= table[other]["elpd_loo"]
    assert table[other]["elpd_diff"] == pytest.approx(table[best]["elpd_loo"] - table[other]["elpd_loo"])
    assert all(row["criterion"] == "grouped_cv" and row["loo_unreliable"] is False for row in table.values())
    assert table[other]["dse_clustered"] > 0
    # The posterior is built on the same ELPD-CV.
    assert posterior["criterion"] == "grouped_cv"
    for name, row in table.items():
        assert posterior["elpd_loo"][name] == pytest.approx(row["elpd_loo"], abs=1e-3)
    # Every training trial was held out exactly once.
    with responses.open() as f:
        n_trials = sum(1 for _ in csv.DictReader(f))
    assert table[best]["n_cv_points"] == n_trials


@pytest.mark.slow
def test_heldout_density_on_the_fits_own_rows_is_its_in_sample_density(models_dir, tmp_path):
    """The held-out density reads the rows it is given: on the training rows
    it equals the log of the posterior-mean likelihood the fit stored."""
    from scipy.special import logsumexp

    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import fit_model

    responses = models_dir / "responses.csv"
    fitted = fit_model("representativeness", models_dir, responses, draws=100, tune=100, chains=1)
    with responses.open() as f:
        rows = list(csv.DictReader(f))
    got = fitted.heldout_lpd(make_stim_data(fitted.model, rows))
    stored = np.asarray(fitted.idata.log_likelihood[fitted.model.observed_RVs[0].name].values)
    stored = stored.reshape(-1, stored.shape[-1])
    np.testing.assert_allclose(got, logsumexp(stored, axis=0) - np.log(stored.shape[0]), rtol=1e-10)
    # And it is not the same on other answers: flipping them changes it.
    flipped = [{**r, "chose_left": str(1 - int(r["chose_left"]))} for r in rows]
    assert not np.allclose(fitted.heldout_lpd(make_stim_data(fitted.model, flipped)), got)


def test_selection_from_config_reads_columns_or_the_stimulus():
    assert Selection.from_config({}) == LOO
    by_set = Selection.from_config({"criterion": "grouped_cv", "cv_group": ["number_set"], "cv_folds": 4})
    assert by_set.cv_group == ("number_set",) and by_set.cv_folds == 4
    assert by_set.as_record()["cv_group"] == ["number_set"]
    with pytest.raises(ValueError, match="cv_group"):
        Selection(criterion="grouped_cv", cv_group="cell")
