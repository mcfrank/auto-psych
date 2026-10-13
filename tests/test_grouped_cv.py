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


def _zoo(tmp_path, names):
    import yaml

    models = tmp_path / "models"
    models.mkdir()
    for n in names:
        (models / f"{n}.py").write_text("")
    (models / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]})
    )
    responses = tmp_path / "responses.csv"
    with responses.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        writer.writeheader()
        for i in range(12):
            writer.writerow({**ROWS[0], "sequence_a": "HT" if i % 2 else "TH", "sequence_b": "H" * 2, "trial_index": str(i), "participant_id": str(i % 6)})
    return models, responses


def _stub_fits(monkeypatch, *, too_slow=(), lpd=None):
    from src.models.pymc_inference import FitTimeLimitExceeded
    from src.pipelines.inner_loop import model_zoo

    calls = {"limited": [], "free": []}

    def fit_model(name, models_dir, path, *, time_limit_sec=None, **kw):
        calls["limited"].append((name, time_limit_sec))
        if name in too_slow:
            raise FitTimeLimitExceeded(name, time_limit_sec, 0.9)

    monkeypatch.setattr(model_zoo, "fit_model", fit_model)
    monkeypatch.setattr(model_zoo, "fit_time_limited_concurrently", lambda *a, **k: None)
    monkeypatch.setattr(
        model_zoo, "fit_models_to_cache", lambda names, *a, **k: calls["free"].extend(names) or {}
    )
    monkeypatch.setattr(
        model_zoo,
        "cv_pointwise",
        lambda names, *a, **k: ({n: np.array([(lpd or {}).get(n, -1.0)]) for n in names}, []),
    )
    return calls


def test_a_model_without_a_finite_cv_score_is_dropped_and_recorded(tmp_path, monkeypatch):
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
    from src.pipelines.inner_loop.model_zoo import _drop_unscorable_cv_models, _manifest_names

    models, responses = _zoo(tmp_path, ["a", "b"])
    _stub_fits(monkeypatch, lpd={"b": -np.inf})
    ledger = HypothesisLedger(tmp_path / "ledger.jsonl")
    selection = Selection(criterion="grouped_cv", cv_group="stimulus", cv_folds=2)
    _drop_unscorable_cv_models(models, responses, selection, ledger=ledger, ledger_context="exp 1")
    assert _manifest_names(models) == ["a"]
    [entry] = ledger.entries()
    assert entry.name == "b" and entry.outcome == "dropped" and "non-finite ELPD-CV" in entry.detail


def test_models_admitted_in_the_loop_fit_their_folds_under_the_time_limit(tmp_path, monkeypatch):
    from src.pipelines.inner_loop.model_zoo import _drop_unscorable_cv_models, _manifest_names

    models, responses = _zoo(tmp_path, ["seed", "fresh", "slow"])
    calls = _stub_fits(monkeypatch, too_slow={"slow"})
    selection = Selection(criterion="grouped_cv", cv_group="stimulus", cv_folds=2)
    _drop_unscorable_cv_models(
        models, responses, selection, limited={"fresh", "slow"}, cache_dir=tmp_path / "cache", time_limit_sec=60
    )
    assert _manifest_names(models) == ["seed", "fresh"]
    assert {name for name, limit in calls["limited"]} == {"fresh", "slow"}
    assert {limit for _, limit in calls["limited"]} == {60}
    assert set(calls["free"]) == {"seed"}  # starting and carried models: no limit


def test_the_cv_screen_does_nothing_under_loo(tmp_path, monkeypatch):
    from src.pipelines.inner_loop.model_zoo import _drop_unscorable_cv_models, _manifest_names

    models, responses = _zoo(tmp_path, ["a"])
    calls = _stub_fits(monkeypatch, lpd={"a": -np.inf})
    _drop_unscorable_cv_models(models, responses, LOO)
    assert _manifest_names(models) == ["a"] and calls == {"limited": [], "free": []}


def test_the_fitted_seed_baseline_picks_its_best_seed_by_the_runs_criterion(tmp_path, monkeypatch):
    """Under grouped CV the baseline's best seed is the ELPD-CV best, as the
    loop's winner is; no PSIS-LOO is computed."""
    from types import SimpleNamespace

    from src.subjective_randomness import holdout_eval

    class Fit:
        def loo_diagnostics(self):
            raise AssertionError("PSIS-LOO computed under grouped CV")

        def convergence_problems(self):
            return []

    (tmp_path / "seed_a.py").write_text("")
    (tmp_path / "seed_b.py").write_text("")
    monkeypatch.setattr(holdout_eval, "fit_model", lambda *a, **k: Fit())
    monkeypatch.setattr(holdout_eval, "_eval_prediction", lambda fitted, rows, **k: np.array([0.2, 0.6, 0.7]))
    monkeypatch.setattr(
        holdout_eval,
        "cv_pointwise",
        lambda names, *a, **k: ({n: np.array([-10.0 if n == "seed_b" else -20.0]) for n in names}, []),
    )
    baseline = holdout_eval._fitted_seed_baseline(
        ["seed_a", "seed_b"],
        tmp_path,
        tmp_path / "r.csv",
        [{}, {}, {}],
        np.array([0.1, 0.5, 0.9]),
        participant_ids=None,
        cache_dir=None,
        fit_kwargs={},
        var_name="p_yes",
        selection=Selection(criterion="grouped_cv", cv_group=("number_set",), cv_folds=3),
    )
    assert baseline["elpd_best_model"] == "seed_b" and baseline["criterion"] == "grouped_cv"
    assert baseline["per_model"]["seed_a"]["elpd_loo"] == -20.0


@pytest.mark.slow
def test_the_admission_sanity_gate_is_the_fits_own_log_likelihood(models_dir):
    """The finite-likelihood gate reads the stored log-likelihood (no PSIS)."""
    from src.models.pymc_inference import fit_model

    fitted = fit_model("representativeness", models_dir, models_dir / "responses.csv", draws=100, tune=100, chains=1)
    stored = np.asarray(fitted.idata.log_likelihood[fitted.model.observed_RVs[0].name].values)
    expected = stored.reshape(-1, stored.shape[-1]).mean(axis=0).sum()
    assert np.isfinite(fitted.mean_log_likelihood())
    assert fitted.mean_log_likelihood() == pytest.approx(expected)
    assert fitted._loo_diagnostics is None
