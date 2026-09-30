"""The novelty gate predicts each model on the pool once per model loop.

The gate compares a candidate's posterior-mean ``p_left`` on the loop's
512-pair novelty pool with every admitted model's. It used to recompute every
admitted model's prediction at every admission, although within one model loop
the data, the pool and every admitted fit are fixed, and the prediction is a
deterministic function of the fit (``predict_p_left`` samples with a fixed
seed). Each prediction runs every posterior draw over the pool, and a model
with a participant effect once per participant. In experiment 1 of the October
2026 live run (40 participants, 3000 draws) most of the set had one, and rounds
grew from 1.5 to 2.9 hours while the job used under 2 of its 16 CPUs.

The loop now keeps each prediction for the rest of the loop, keyed by the fit
it comes from (the in-process fit key: model source, responses file, sampler
settings), the pool and the participants, so new data or a changed model file
is predicted afresh. The verdicts are those of recomputing.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.models import pymc_inference as pi
from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from tests.inner_loop_fixtures import write_responses, write_seed_models
from tests.test_concurrent_candidate_fits import _admit_in_order
from tests.test_parallel_candidates import _patch_loop_internals

_REAL_MIN_PREDICTION_RMSE = model_zoo._min_prediction_rmse


@pytest.fixture(autouse=True)
def _clean_fit_cache():
    pi.clear_fit_cache()
    yield
    pi.clear_fit_cache()


def _counting_pool_prediction(monkeypatch):
    """Replace the pool prediction with a constant per model (distinct models,
    so every candidate is novel) and count the calls by model name."""
    calls = Counter()
    level = {}

    def predict(fitted, pool_rows, *, model_name, participant_ids):
        calls[model_name] += 1
        value = level.setdefault(model_name, 0.1 + 0.05 * len(level))
        return np.full(len(pool_rows), value)

    monkeypatch.setattr(model_zoo, "_pool_prediction", predict)
    return calls


def test_a_model_loop_predicts_each_model_on_the_pool_once(tmp_path, monkeypatch):
    _patch_loop_internals(monkeypatch)
    # The real gate, over stub fits; only the pool prediction is counted.
    monkeypatch.setattr(model_zoo, "_min_prediction_rmse", _REAL_MIN_PREDICTION_RMSE)
    calls = _counting_pool_prediction(monkeypatch)

    def spawn(candidate_dir, docs, **kwargs):
        name = f"m_{candidate_dir.parent.name.replace('_', '')}_{candidate_dir.name}"
        (candidate_dir / "candidate.py").write_text(f"# {name}\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(f"People do {name}.\n", encoding="utf-8")
        (candidate_dir / "model_name.txt").write_text(name, encoding="utf-8")
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)
    pymc_orchestrator.run_pymc_inner_loop(
        responses_path=write_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
        max_iterations=2,
        candidate_count=2,
        enable_critique=False,
        fit_kwargs={},
    )

    ledger = HypothesisLedger(tmp_path / "model_loop" / "attempted_hypotheses.jsonl")
    admitted = [e.name for e in ledger.entries() if e.outcome == "admitted"]
    assert len(admitted) == 4  # two rounds of two, so later admissions compare with earlier ones
    # Every model predicted once: the seed and each candidate, not again at
    # every later admission (the seed used to be predicted four times).
    assert calls == Counter({name: 1 for name in ["model_a", *admitted]})


def _set_with_candidate(tmp_path):
    """An admitted seed and a staged candidate, as the gate finds them."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in ("seed", "cand"):
        (models_dir / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "the seed"}]}),
        encoding="utf-8",
    )
    responses = tmp_path / "responses.csv"
    responses.write_text(
        "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
        "HHTT,HTHT,1,0,1\nHHTT,HTHT,2,0,0\n",
        encoding="utf-8",
    )
    return models_dir, responses


def _gate(models_dir, responses, predictions, pool):
    return _REAL_MIN_PREDICTION_RMSE(
        "cand", models_dir, responses, pool_rows=pool, fit_kwargs={}, predictions=predictions
    )


def test_saved_predictions_give_the_same_nearest_model_and_rmse(tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: SimpleNamespace())
    calls = _counting_pool_prediction(monkeypatch)
    models_dir, responses = _set_with_candidate(tmp_path)
    pool = model_zoo.novelty_pool_rows()

    recomputed = _gate(models_dir, responses, None, pool)
    saved = {}
    first = _gate(models_dir, responses, saved, pool)
    again = _gate(models_dir, responses, saved, pool)

    assert recomputed == first == again == ("seed", pytest.approx(0.05))
    # One pass each without the dict, one each for the first saved call, none after.
    assert calls == Counter({"seed": 2, "cand": 2})


def test_new_data_a_changed_model_or_another_pool_is_predicted_afresh(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: SimpleNamespace())
    calls = _counting_pool_prediction(monkeypatch)
    models_dir, responses = _set_with_candidate(tmp_path)
    pool = model_zoo.novelty_pool_rows()
    saved = {}

    _gate(models_dir, responses, saved, pool)
    assert calls == Counter({"seed": 1, "cand": 1})

    # New data: every fit is new, so every prediction is.
    with responses.open("a", encoding="utf-8") as f:
        f.write("HTTH,THHT,3,0,1\n")
    _gate(models_dir, responses, saved, pool)
    assert calls == Counter({"seed": 2, "cand": 2})

    # A changed model file (a repair under the same name, say).
    (models_dir / "cand.py").write_text("# cand, repaired\n", encoding="utf-8")
    _gate(models_dir, responses, saved, pool)
    assert calls == Counter({"seed": 2, "cand": 3})

    # Another pool.
    _gate(models_dir, responses, saved, pool[:100])
    assert calls == Counter({"seed": 3, "cand": 4})


@pytest.mark.slow
def test_saved_predictions_change_no_admission_verdict(tmp_path, monkeypatch):
    recomputed = _admit_in_order(tmp_path / "recomputed", prefit=True)

    calls = Counter()
    real_pool_prediction = model_zoo._pool_prediction

    def counting(fitted, pool_rows, *, model_name, participant_ids):
        calls[model_name] += 1
        return real_pool_prediction(
            fitted, pool_rows, model_name=model_name, participant_ids=participant_ids
        )

    monkeypatch.setattr(model_zoo, "_pool_prediction", counting)
    saved = _admit_in_order(tmp_path / "saved", prefit=True, novelty_predictions={})

    assert saved["verdicts"] == recomputed["verdicts"]
    assert saved["manifest"] == recomputed["manifest"]
    assert saved["ledger"] == recomputed["ledger"]  # the same RMSEs in every reason
    assert saved["fits"] == recomputed["fits"]
    # The wave compared later candidates with earlier admissions, and no model
    # was predicted twice.
    assert calls["bayesian_fair_coin"] == 1 and calls["rep_a"] == 1
    assert max(calls.values()) == 1
