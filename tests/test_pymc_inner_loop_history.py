"""Unit tests for per-step history tracking in the PyMC inner loop.

The loop scores the model set once after seeding and once after each candidate
round; every scoring step must be recorded to ``history.json`` (so a crashed
run keeps a partial trajectory) and returned under ``"history"``. MCMC and
agent spawning are stubbed — these tests cover only the bookkeeping.
"""

from __future__ import annotations

import json


import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
import src.pipelines.inner_loop.scoring as scoring
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import (
    canned_posterior,
    write_responses,
    write_seed_models,
)


def _patch_scoring(monkeypatch, posteriors_per_call):
    calls = {"n": 0}

    def fake_model_posterior(responses_path, models_dir, **kwargs):
        result = posteriors_per_call[calls["n"]]
        calls["n"] += 1
        return result

    monkeypatch.setattr(scoring, "model_posterior", fake_model_posterior)
    monkeypatch.setattr(scoring, "compare_table", lambda *args, **kwargs: {})
    # _prune_losers looks up compare_table in model_zoo's namespace:
    monkeypatch.setattr(model_zoo, "compare_table", lambda *args, **kwargs: {})
    # Stub fittability so the fake stub seed models are not dropped/scored as
    # un-fittable (they are not real PyMC models). These functions are looked up
    # in model_zoo's namespace (where _drop_unfittable_models etc. now live).
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    # Candidate admission now ends with a real MCMC fit-gate; stub it so the fake
    # stub candidates (not real PyMC models) are admitted without sampling.
    # The stub fit is not a real trace: pass the convergence gate.
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: [])
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    # The experiment-start screen samples the whole set in one batch; no MCMC here.
    monkeypatch.setattr(model_zoo, "fit_models_to_cache", lambda names, *a, **k: {})
    # Admission also gates on a finite ELPD-LOO; stub it finite for stub candidates.
    monkeypatch.setattr(model_zoo, "log_likelihood", lambda *a, **k: -100.0)
    # Novelty gate is covered by test_novelty_gate.py; neutralize it here.
    monkeypatch.setattr(
        model_zoo,
        "_min_prediction_rmse",
        lambda *a, **k: (None, float("inf")),
    )


def _patch_candidates(monkeypatch):
    """Each spawned agent writes a candidate; validation is stubbed to accept."""

    def fake_spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            "People use heuristic H.\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", fake_spawn)
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, models_dir: object())


def test_inner_loop_writes_history_entry_per_scoring_step(tmp_path, monkeypatch):
    _patch_scoring(
        monkeypatch,
        [
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
            canned_posterior(
                "iter1_candidate0", ["model_a", "model_b", "iter0_candidate0"]
            ),
        ],
    )
    _patch_candidates(monkeypatch)

    result = run_pymc_inner_loop(
        write_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=2,
        candidate_count=1,
    )

    history = result["history"]
    assert [entry["step"] for entry in history] == [0, 1, 2]
    assert [entry["iteration"] for entry in history] == [None, 0, 1]
    assert [entry["best_model"] for entry in history] == [
        "model_a",
        "iter0_candidate0",
        "iter1_candidate0",
    ]
    for entry in history:
        assert set(entry["posteriors"]) == set(entry["elpd_loo"])

    history_path = tmp_path / "results" / "history.json"
    assert result["history_path"] == str(history_path)
    assert json.loads(history_path.read_text(encoding="utf-8")) == history


def test_inner_loop_history_seed_only_has_single_step(tmp_path, monkeypatch):
    _patch_scoring(monkeypatch, [canned_posterior("model_b", ["model_a"])])

    result = run_pymc_inner_loop(
        write_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=0,
    )

    history = json.loads(
        (tmp_path / "results" / "history.json").read_text(encoding="utf-8")
    )
    assert history == result["history"]
    assert len(history) == 1
    assert history[0]["step"] == 0
    assert history[0]["iteration"] is None
    assert history[0]["best_model"] == "model_b"


def _row(rank, elpd_loo, unreliable=False):
    return {
        "rank": rank,
        "elpd_loo": elpd_loo,
        "elpd_diff": 0.0,
        "dse": 0.0,
        "weight": 0.5,
        "loo_unreliable": unreliable,
    }


def test_history_best_model_follows_the_export_rule(tmp_path, monkeypatch):
    """history.json's per-step ``best_model`` is what the trajectory evaluation
    scores, and the export is what the next experiment carries; they must be
    the same model. Both follow ELPD rank among reliable models — never the
    rounded posterior argmax, which here is an unreliable fit."""
    posteriors = [
        canned_posterior("model_a", ["model_b"]),
        canned_posterior("iter0_candidate0", ["model_a", "model_b"]),
    ]
    _patch_scoring(monkeypatch, posteriors)
    _patch_candidates(monkeypatch)

    def fake_compare(responses_path, models_dir, **kwargs):
        # The posterior argmax at every step is unreliable; model_b is the
        # ELPD-best reliable model although its posterior is never the largest.
        names = pymc_orchestrator._manifest_names(models_dir)
        rows = {}
        for name in names:
            if name == "model_b":
                rows[name] = _row(1, -11.0)
            elif name == "model_a":
                candidate_admitted = "iter0_candidate0" in names
                rows[name] = _row(
                    2 if candidate_admitted else 0,
                    -12.0,
                    unreliable=not candidate_admitted,
                )
            else:
                rows[name] = _row(0, -10.0, unreliable=True)
        return rows

    monkeypatch.setattr(scoring, "compare_table", fake_compare)
    # Pruning would call compare_table too; keep the round to selection only.
    monkeypatch.setattr(pymc_orchestrator, "_prune_losers", lambda *a, **k: [])

    result = run_pymc_inner_loop(
        write_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=1,
        candidate_count=1,
    )

    history = json.loads(
        (tmp_path / "results" / "history.json").read_text(encoding="utf-8")
    )
    assert [entry["best_model"] for entry in history] == ["model_b", "model_b"]
    assert [entry["argmax_model"] for entry in history] == [
        "model_a",
        "iter0_candidate0",
    ]
    assert history[0]["excluded_unreliable"] == ["model_a"]
    assert history[1]["excluded_unreliable"] == ["iter0_candidate0"]
    assert result["best_model"] == history[-1]["best_model"]


def test_record_history_step_stores_the_round_critique_status(tmp_path):
    """Every round's history entry carries the critique status the round ran
    with (statistics proposed / significant, "no critique", or disabled), so
    an absent critique is visible in the run record rather than only in the
    log. The seed scoring step precedes any round and has none."""
    from src.pipelines.inner_loop.scoring import _record_history_step

    history: list = []
    posterior = canned_posterior("model_a", ["model_b"])
    _record_history_step(history, tmp_path, posterior, {}, iteration=None)
    assert "critique" not in history[0]

    status = {
        "status": "critiqued",
        "incumbent": "model_a",
        "attempts": 1,
        "n_statistics": 8,
        "n_significant": 2,
        "n_significant_fdr": 1,
    }
    _record_history_step(history, tmp_path, posterior, {}, iteration=0, critique=status)
    assert history[1]["critique"] == status
    on_disk = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert on_disk[1]["critique"] == status

    import pytest

    with pytest.raises(ValueError, match="critique"):
        _record_history_step(history, tmp_path, posterior, {}, iteration=1)
    with pytest.raises(ValueError, match="critique"):
        _record_history_step(
            history, tmp_path, posterior, {}, iteration=None, critique=status
        )
