"""A design after data reuses the previous model loop's fits; it never samples.

The design for experiment N+1 predicts each carried model from its posterior
given all data so far: the previous experiment's cumulative
``model_loop/responses.csv``. That loop had already fitted every carried model
on exactly that file (the carried set is its survivors) and cached the fits,
but the design refitted them at cheaper design-time settings (500 tune / 500
draws / 2 chains), into a cache of its own, with no convergence gate: an extra
MCMC fit of every carried model per experiment, of a posterior that could
differ from the one the loop scored, pruned and exported (user decision
2026-10-08).

Now the design is given the loop's fit cache and sampler settings and loads
each model's fit from it (``load_cached_fit``: the fit ``fit_model`` returned
to the loop, its near-miss refit included). A fit missing from the cache
raises ``FitNotCached``; nothing is sampled.
"""

from __future__ import annotations

import shutil

import pytest

import src.models.pymc_inference as pymc_inference
from src.models.pymc_inference import FitNotCached, fit_model, load_cached_fit
from src.pipelines.outer_loop import eig as eig_mod
from src.pipelines.outer_loop.orchestrator import run_design_programmatic
from src.tasks.subjective_randomness import SUBJECTIVE_RANDOMNESS
from tests.paths import PYMC_MODEL_FIXTURES_DIR, sr_pairs

LOOP_SETTINGS = {"draws": 60, "tune": 60, "chains": 2, "target_accept": 0.8, "random_seed": 3}


def _models(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(PYMC_MODEL_FIXTURES_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(
        PYMC_MODEL_FIXTURES_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml"
    )
    return models_dir


def _never_sample(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the design sampled a fit")

    monkeypatch.setattr(pymc_inference, "fit_model", refuse)
    monkeypatch.setattr(pymc_inference._import_pymc(), "sample", refuse)


def test_a_fit_missing_from_the_cache_raises_instead_of_sampling(tmp_path, monkeypatch):
    models_dir = _models(tmp_path)
    _never_sample(monkeypatch)
    with pytest.raises(FitNotCached, match="bayesian_fair_coin"):
        load_cached_fit(
            "bayesian_fair_coin",
            models_dir,
            PYMC_MODEL_FIXTURES_DIR / "responses.csv",
            cache_dir=tmp_path / "fit_cache",
            **LOOP_SETTINGS,
        )


@pytest.mark.slow
def test_the_cached_fit_is_the_one_fit_model_returned_and_its_refit_is_required(
    tmp_path, monkeypatch
):
    models_dir = _models(tmp_path)
    responses = PYMC_MODEL_FIXTURES_DIR / "responses.csv"
    cache = tmp_path / "fit_cache"
    # Only the first fit is cached (a fit this short can be a near miss).
    monkeypatch.setattr(pymc_inference, "_refit_decision", lambda name, f, s: False)
    fitted = fit_model("bayesian_fair_coin", models_dir, responses, cache_dir=cache, **LOOP_SETTINGS)
    files = set(cache.iterdir())

    _never_sample(monkeypatch)
    loaded = load_cached_fit(
        "bayesian_fair_coin", models_dir, responses, cache_dir=cache, **LOOP_SETTINGS
    )
    assert loaded.fingerprint == fitted.fingerprint
    assert set(cache.iterdir()) == files

    # A near miss's fit is its refit: when that is not cached either, raise.
    monkeypatch.setattr(pymc_inference, "_refit_decision", lambda name, f, s: True)
    with pytest.raises(FitNotCached, match="refit"):
        load_cached_fit(
            "bayesian_fair_coin", models_dir, responses, cache_dir=cache, **LOOP_SETTINGS
        )


def test_the_design_loads_each_model_with_the_loops_cache_and_settings(tmp_path, monkeypatch):
    import src.models.data_binding as data_binding

    loaded = []

    class Fitted:
        model = None

        def predict_p_left_draws(self, stim_data, *, seed, max_draws, var_name="p_left"):
            return "draws"

    def fake_load(name, models_dir, responses_csv, *, cache_dir, **settings):
        loaded.append((name, cache_dir, responses_csv, settings))
        return Fitted()

    _never_sample(monkeypatch)
    monkeypatch.setattr(pymc_inference, "load_cached_fit", fake_load)
    monkeypatch.setattr(data_binding, "make_stim_data", lambda model, rows: {})
    responses = tmp_path / "experiment1" / "model_loop" / "responses.csv"
    cache = tmp_path / "experiment1" / "model_loop" / ".fit_cache"

    draws, screened = eig_mod._posterior_p_left_draws(
        ["careful", "plain"],
        tmp_path,
        [],
        responses_csv=responses,
        fit_cache_dir=cache,
        fit_kwargs=LOOP_SETTINGS,
        max_draws=10,
        seed=0,
        task=SUBJECTIVE_RANDOMNESS,
    )

    assert loaded == [
        ("careful", cache, responses, LOOP_SETTINGS),
        ("plain", cache, responses, LOOP_SETTINGS),
    ]
    assert draws == {"careful": "draws", "plain": "draws"} and screened == []


def test_a_design_after_data_without_the_loops_fits_raises(tmp_path, monkeypatch):
    models_dir = _models(tmp_path)
    _never_sample(monkeypatch)
    with pytest.raises(FitNotCached):
        eig_mod.design_exhaustive(
            models_dir,
            task=SUBJECTIVE_RANDOMNESS,
            pool=sr_pairs(3),
            n_select=2,
            n_samples=5,
            n_scenarios=10,
            n_responses=4,
            responses_csv=PYMC_MODEL_FIXTURES_DIR / "responses.csv",
            fit_cache_dir=tmp_path / "empty_cache",
            fit_kwargs=LOOP_SETTINGS,
        )


@pytest.mark.parametrize("missing", ["fit_cache_dir", "fit_kwargs"])
def test_a_design_after_data_must_be_told_the_loops_cache_and_settings(tmp_path, missing):
    given = {"fit_cache_dir": tmp_path / "cache", "fit_kwargs": LOOP_SETTINGS}
    del given[missing]
    with pytest.raises(ValueError, match=missing):
        eig_mod.design_exhaustive(
            _models(tmp_path),
            task=SUBJECTIVE_RANDOMNESS,
            pool=sr_pairs(3),
            n_select=2,
            n_responses=4,
            responses_csv=PYMC_MODEL_FIXTURES_DIR / "responses.csv",
            **given,
        )


def _previous_experiment(tmp_path):
    prev = tmp_path / "experiment1"
    (prev / "model_loop").mkdir(parents=True)
    (prev / "model_loop" / "responses.csv").write_text("participant_id\n", encoding="utf-8")
    exp_dir = tmp_path / "experiment2"
    exp_dir.mkdir()
    return prev, exp_dir


def test_the_design_stage_threads_the_previous_loops_cache_and_settings(tmp_path, monkeypatch):
    calls = {}

    def fake_design(models_dir, registry_path=None, **kwargs):
        calls.update(kwargs)
        return [{"sequence_a": "HHH", "sequence_b": "HTH", "eig": 0.5,
                 "selection_rank": 1, "joint_eig_bits": 0.5}]

    monkeypatch.setattr("src.pipelines.outer_loop.eig.design_exhaustive", fake_design)
    prev, exp_dir = _previous_experiment(tmp_path)
    cache = prev / "model_loop" / ".fit_cache"

    run_design_programmatic(
        exp_dir, "subjective_randomness", exp_num=2, prev_exp_dir=prev, k=1,
        n_responses=40, fit_kwargs=LOOP_SETTINGS, loop_fit_cache_dir=cache,
    )

    assert calls["responses_csv"] == prev / "model_loop" / "responses.csv"
    assert calls["fit_cache_dir"] == cache
    assert calls["fit_kwargs"] == LOOP_SETTINGS


def test_the_design_stage_after_data_refuses_without_the_loops_cache(tmp_path):
    prev, exp_dir = _previous_experiment(tmp_path)
    with pytest.raises(ValueError, match="loop_fit_cache_dir"):
        run_design_programmatic(
            exp_dir, "subjective_randomness", exp_num=2, prev_exp_dir=prev, k=1,
            n_responses=40, fit_kwargs=LOOP_SETTINGS,
        )


def test_run_py_designs_from_the_previous_experiments_loop_fits(tmp_path, monkeypatch):
    """run.py hands the design the previous experiment's model_loop/.fit_cache
    (where its loop cached every fit) and the run's sampler settings."""
    import src.pipelines.outer_loop.run as outer_run

    seen = {}
    monkeypatch.setattr(
        outer_run, "run_design_programmatic", lambda exp_dir, project_id, **kw: seen.update(kw)
    )
    prev = tmp_path / "experiment1"
    prev.mkdir()
    outer_run._run_design_stage(
        tmp_path / "experiment2", "subjective_randomness", exp_num=2, prev_exp_dir=prev,
        n_participants=40, fit_kwargs=LOOP_SETTINGS,
    )
    assert seen["loop_fit_cache_dir"] == prev / "model_loop" / ".fit_cache"
    assert seen["fit_kwargs"] == LOOP_SETTINGS
    assert seen["n_responses"] == 40
