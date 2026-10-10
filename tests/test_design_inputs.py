"""What the stimulus design is computed from.

- Experiments >= 2 predict from the models' posteriors given ALL data
  collected so far (the previous experiment's cumulative inner-loop
  responses) — not the previous experiment alone. Those are the previous
  model loop's own fits, loaded from its cache
  (tests/test_design_reuses_loop_fits.py).
- The EIG scores each stimulus as answered by every participant of the
  experiment; the caller must say how many there are.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.outer_loop import eig as eig_mod
from src.pipelines.outer_loop.orchestrator import run_design_programmatic
from tests.paths import PYMC_MODEL_FIXTURES_DIR
from src.tasks.subjective_randomness import SUBJECTIVE_RANDOMNESS
from tests.paths import sr_pairs

LOOP_FITS = {"fit_kwargs": {"draws": 10}}


def _loop_fits(prev):
    """The previous loop's fit cache and settings, as run.py passes them."""
    return {**LOOP_FITS, "loop_fit_cache_dir": prev / "model_loop" / ".fit_cache"}


@pytest.fixture
def captured_design(monkeypatch):
    captured = {}

    def fake_design_exhaustive(models_dir, registry_path=None, **kwargs):
        captured.update(kwargs, registry_path=registry_path)
        return [{"sequence_a": "HT", "sequence_b": "TH"}]

    monkeypatch.setattr(eig_mod, "design_exhaustive", fake_design_exhaustive)
    return captured


def test_later_designs_fit_on_all_data_collected_so_far(tmp_path, captured_design):
    prev, exp = tmp_path / "experiment1", tmp_path / "experiment2"
    run_design_programmatic(
        exp, "subjective_randomness", exp_num=2, prev_exp_dir=prev, k=4, n_responses=40,
        **_loop_fits(prev),
    )
    assert captured_design["responses_csv"] == prev / "model_loop" / "responses.csv"


@pytest.mark.parametrize("exp_num", [1, 2])
def test_the_eig_counts_every_participants_response(tmp_path, captured_design, exp_num):
    prev = tmp_path / "experiment1" if exp_num > 1 else None
    run_design_programmatic(
        tmp_path / f"experiment{exp_num}",
        "subjective_randomness",
        exp_num=exp_num,
        prev_exp_dir=prev,
        k=4,
        n_responses=40,
        **(_loop_fits(prev) if prev is not None else {}),
    )
    assert captured_design["n_responses"] == 40


def test_the_caller_must_say_how_many_responses_each_stimulus_gets(tmp_path):
    with pytest.raises(TypeError, match="n_responses"):
        run_design_programmatic(tmp_path / "experiment1", "subjective_randomness", k=4)


def test_a_saturated_selection_is_filled_by_single_response_eig(tmp_path, monkeypatch):
    """The 40-response selection stops at its noise floor; the remaining slots
    are filled by single-response EIG, conditioned on the picks so far, and
    every stimulus says which objective chose it."""
    import src.models.eig_selection as eig_selection
    import src.models.pymc_inference as pymc_inference
    from src.models.eig_selection import JointEIGSelection

    calls = []

    def fake_select(draws, n_select, **kwargs):
        calls.append({"n_select": n_select, **kwargs})
        if kwargs["n_responses"] == 40:
            return JointEIGSelection(
                [3, 7], [1.0, 1.5], 10, stopped_at_noise_floor=True
            )
        return JointEIGSelection([1, 2, 5], [0.4, 0.6, 0.7], 10)

    monkeypatch.setattr(eig_selection, "select_n_joint_eig", fake_select)
    monkeypatch.setattr(
        pymc_inference,
        "prior_predict_p_left_draws",
        lambda names, d, rows, **k: {n: np.full((4, len(rows)), 0.5) for n in names},
    )
    monkeypatch.setattr(eig_mod, "_load_model_names", lambda d: ["m1", "m2"])
    monkeypatch.setattr(
        eig_mod, "_screen_usable_models", lambda names, d, rows, task, **k: (names, [])
    )

    stimuli = eig_mod.design_exhaustive(
        tmp_path, task=SUBJECTIVE_RANDOMNESS, pool=sr_pairs(2,3), n_select=5, n_responses=40, seed=9
    )

    assert calls[0]["stop_below_noise"] is True and calls[0]["n_responses"] == 40
    assert calls[1]["n_select"] == 3 and calls[1]["n_responses"] == 1
    assert list(calls[1]["preselected"]) == [3, 7]
    assert [s["source"] for s in stimuli] == ["eig"] * 2 + [
        "eig_single_response_fill"
    ] * 3
    assert [s["selection_rank"] for s in stimuli] == [1, 2, 3, 4, 5]


def test_the_design_search_settings_reach_both_selections(tmp_path, monkeypatch):
    """The lazy/exact search, the scoring precision and the thread count are
    passed to the noise-floor selection and to the fill alike; the threads
    default to the CPUs the process is allocated."""
    import src.models.eig_selection as eig_selection
    import src.models.pymc_inference as pymc_inference
    from src.models.eig_selection import JointEIGSelection

    calls = []

    def fake_select(draws, n_select, **kwargs):
        calls.append(kwargs)
        if kwargs["n_responses"] == 40:
            return JointEIGSelection([3], [1.0], 10, stopped_at_noise_floor=True)
        return JointEIGSelection([1], [0.4], 10)

    monkeypatch.setattr(eig_selection, "select_n_joint_eig", fake_select)
    monkeypatch.setattr(pymc_inference, "allocated_cpus", lambda: 7)
    monkeypatch.setattr(
        pymc_inference,
        "prior_predict_p_left_draws",
        lambda names, d, rows, **k: {n: np.full((4, len(rows)), 0.5) for n in names},
    )
    monkeypatch.setattr(eig_mod, "_load_model_names", lambda d: ["m1", "m2"])
    monkeypatch.setattr(
        eig_mod, "_screen_usable_models", lambda names, d, rows, task, **k: (names, [])
    )

    eig_mod.design_exhaustive(
        tmp_path,
        task=SUBJECTIVE_RANDOMNESS, pool=sr_pairs(2,3),
        n_select=2,
        n_responses=40,
        seed=9,
        lazy=True,
        scoring_dtype="float32",
        leave_one_out=False,
    )

    assert len(calls) == 2
    for call in calls:
        assert call["lazy"] is True and call["dtype"] == "float32"
        assert call["leave_one_out"] is False
        assert call["n_threads"] == 7
        assert call["lazy_batch_size"] == eig_mod.DESIGN_LAZY_BATCH_SIZE
        assert call["refresh_every"] == eig_mod.DESIGN_REFRESH_EVERY


def test_the_design_searches_lazily_in_float32_by_default(tmp_path, monkeypatch):
    """Validated against exact float64 greedy on two experiment-2 designs
    (scripts/subjective_randomness/validate_lazy_eig.py)."""
    import src.models.eig_selection as eig_selection
    import src.models.pymc_inference as pymc_inference
    from src.models.eig_selection import JointEIGSelection

    calls = []

    def fake_select(draws, n_select, **kwargs):
        calls.append(kwargs)
        return JointEIGSelection([3, 4], [1.0, 1.2], 10)

    monkeypatch.setattr(eig_selection, "select_n_joint_eig", fake_select)
    monkeypatch.setattr(
        pymc_inference,
        "prior_predict_p_left_draws",
        lambda names, d, rows, **k: {n: np.full((4, len(rows)), 0.5) for n in names},
    )
    monkeypatch.setattr(eig_mod, "_load_model_names", lambda d: ["m1", "m2"])
    monkeypatch.setattr(
        eig_mod, "_screen_usable_models", lambda names, d, rows, task, **k: (names, [])
    )

    eig_mod.design_exhaustive(
        tmp_path, task=SUBJECTIVE_RANDOMNESS, pool=sr_pairs(2,3), n_select=2, n_responses=40, seed=9
    )

    [call] = calls
    assert call["lazy"] is True and call["dtype"] == "float32"
    assert (call["lazy_batch_size"], call["refresh_every"]) == (512, 16)
    # Each scenario's generating draw is left out of its likelihood average.
    assert call["leave_one_out"] is True


def _seed_experiment_models(exp_dir):
    """Copy the fixture PyMC models into exp_dir/cognitive_models."""
    import shutil

    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in (
        "bayesian_fair_coin.py",
        "representativeness.py",
        "models_manifest.yaml",
    ):
        shutil.copyfile(PYMC_MODEL_FIXTURES_DIR / name, models_dir / name)
    return models_dir


@pytest.mark.slow
def test_run_design_programmatic_writes_stimuli_json(tmp_path):
    """Experiment 1: exhaustive design scores the experiment's ACTUAL PyMC
    model set from its prior predictive (no pure-Python twins involved)."""
    import json

    exp_dir = tmp_path / "experiment1"
    _seed_experiment_models(exp_dir)
    run_design_programmatic(
        exp_dir, "subjective_randomness", k=4, pool=sr_pairs(3,4), n_responses=40
    )
    data = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 4
    assert all("sequence_a" in s and "sequence_b" in s for s in data)
    assert all("eig" in s and "selection_rank" in s for s in data)


def test_run_design_programmatic_posterior_wiring(tmp_path, monkeypatch):
    """Experiment >= 2: the previous experiment's responses and registry are
    threaded into design_exhaustive's posterior mode, with the previous
    loop's fit cache and sampler settings."""
    import json

    calls: dict = {}

    def fake_design(models_dir, registry_path=None, **kwargs):
        calls["models_dir"] = models_dir
        calls["registry_path"] = registry_path
        calls.update(kwargs)
        return [
            {
                "sequence_a": "HHH",
                "sequence_b": "HTH",
                "eig": 0.5,
                "selection_rank": 1,
                "joint_eig_bits": 0.5,
            }
        ]

    monkeypatch.setattr("src.pipelines.outer_loop.eig.design_exhaustive", fake_design)

    prev = tmp_path / "experiment1"
    (prev / "model_loop").mkdir(parents=True)
    (prev / "model_loop" / "responses.csv").write_text(
        "participant_id\n", encoding="utf-8"
    )
    exp_dir = tmp_path / "experiment2"
    exp_dir.mkdir()

    run_design_programmatic(
        exp_dir,
        "subjective_randomness",
        exp_num=2,
        prev_exp_dir=prev,
        k=1,
        pool=sr_pairs(3,4),
        n_responses=40,
        **_loop_fits(prev),
    )

    assert calls["models_dir"] == exp_dir / "cognitive_models"
    assert calls["registry_path"] == prev / "model_registry.yaml"
    # All data so far: the previous experiment's cumulative inner-loop responses.
    assert calls["responses_csv"] == prev / "model_loop" / "responses.csv"
    # The previous loop's own fits: its cache and sampler settings.
    assert calls["fit_cache_dir"] == prev / "model_loop" / ".fit_cache"
    assert calls["fit_kwargs"] == LOOP_FITS["fit_kwargs"]
    assert calls["n_select"] == 1
    data = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert data[0]["eig"] == 0.5
