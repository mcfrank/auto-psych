"""What the stimulus design is computed from.

- Experiments >= 2 fit the design's models on ALL data collected so far (the
  previous experiment's cumulative inner-loop responses), matching the
  posterior the inner loop reasons with — not the previous experiment alone.
- Design-time fits use the model's own declared target_accept, else 0.9 (a
  compromise between the production 0.99 and PyMC's 0.8).
- The EIG scores each stimulus as answered by every participant of the
  experiment; the caller must say how many there are.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.outer_loop import eig as eig_mod
from src.pipelines.outer_loop.orchestrator import run_design_programmatic


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
        exp, "subjective_randomness", exp_num=2, prev_exp_dir=prev, k=4, n_responses=40
    )
    assert captured_design["responses_csv"] == prev / "model_loop" / "responses.csv"


@pytest.mark.parametrize("exp_num", [1, 2])
def test_the_eig_counts_every_participants_response(tmp_path, captured_design, exp_num):
    prev = tmp_path / "experiment1" if exp_num > 1 else None
    run_design_programmatic(
        tmp_path / f"experiment{exp_num}", "subjective_randomness",
        exp_num=exp_num, prev_exp_dir=prev, k=4, n_responses=40,
    )
    assert captured_design["n_responses"] == 40


def test_the_caller_must_say_how_many_responses_each_stimulus_gets(tmp_path):
    with pytest.raises(TypeError, match="n_responses"):
        run_design_programmatic(tmp_path / "experiment1", "subjective_randomness", k=4)


def test_design_fits_use_the_models_own_target_accept_else_0_9(tmp_path, monkeypatch):
    import src.models.data_binding as data_binding
    import src.models.pymc_inference as pymc_inference

    requested = {}

    class Fitted:
        model = None

        def predict_p_left_draws(self, stim_data, *, seed, max_draws):
            return "draws"

    def fake_fit_model(name, models_dir, responses_csv, **kwargs):
        requested[name] = kwargs["target_accept"]
        return Fitted()

    declared = {"careful": {"target_accept": 0.97}, "plain": {}}
    monkeypatch.setattr(pymc_inference, "fit_model", fake_fit_model)
    monkeypatch.setattr(pymc_inference, "model_sampler_settings", lambda n, d: declared[n])
    monkeypatch.setattr(data_binding, "make_stim_data", lambda model, rows: {})
    eig_mod._posterior_p_left_draws(
        ["careful", "plain"], tmp_path, [], responses_csv=tmp_path / "r.csv",
        fit_cache_dir=None, max_draws=10, seed=0,
    )
    assert requested == {"careful": 0.97, "plain": 0.9}


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
            return JointEIGSelection([3, 7], [1.0, 1.5], 10, stopped_at_noise_floor=True)
        return JointEIGSelection([1, 2, 5], [0.4, 0.6, 0.7], 10)

    monkeypatch.setattr(eig_selection, "select_n_joint_eig", fake_select)
    monkeypatch.setattr(
        pymc_inference, "prior_predict_p_left_draws",
        lambda names, d, rows, **k: {n: np.full((4, len(rows)), 0.5) for n in names},
    )
    monkeypatch.setattr(eig_mod, "_load_model_names", lambda d: ["m1", "m2"])
    monkeypatch.setattr(eig_mod, "_screen_usable_models", lambda names, d, row: (names, []))

    stimuli = eig_mod.design_exhaustive(
        tmp_path, lengths=(2, 3), n_select=5, n_responses=40, seed=9
    )

    assert calls[0]["stop_below_noise"] is True and calls[0]["n_responses"] == 40
    assert calls[1]["n_select"] == 3 and calls[1]["n_responses"] == 1
    assert list(calls[1]["preselected"]) == [3, 7]
    assert [s["source"] for s in stimuli] == ["eig"] * 2 + ["eig_single_response_fill"] * 3
    assert [s["selection_rank"] for s in stimuli] == [1, 2, 3, 4, 5]
