"""The run's starting models are treated like every other model.

They used to be protected: never pruned, never retired by the live-set cap,
always carried to the next experiment, and a starting model that could not be
fitted stopped the cell. Now (user decision, 2026-09-28) a starting model is
pruned by the same statistical rule, retired by the same cap ordering,
dropped and recorded when it cannot be fitted, and, once gone, leaves the
carried set and the next design's prior and appears in the ledger and the
refinement menu like any pruned model. What stays: no candidate may take a
starting model's name, the fitted starting-model baseline covers every
starting model (its file is found in ``pruned/`` once pruned), and the run
records that its starting models were prunable, so sweeps run before and
after the change can be told apart.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
from src.pipelines.inner_loop.candidate_agent import _write_refinement_menu
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME, HypothesisLedger
from src.pipelines.inner_loop.model_zoo import (
    _cap_live_set,
    _drop_nonfinite_elpd_models,
    _drop_unfittable_models,
    _prune_losers,
)
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop.model_loop_runner import (
    STARTING_MODELS_FILENAME,
    _export_inner_loop_models,
    run_starting_models,
    starting_models_prunable,
    update_registry_from_interpretation,
)
from src.pipelines.outer_loop.orchestrator import carry_forward_cognitive_models
from src.subjective_randomness import holdout_eval
from tests.inner_loop_fixtures import write_responses, write_seed_models
from tests.test_pymc_inner_loop_ledger import _ledger_rows, _manifest_names, _patch_scoring

MODEL_SRC = "import pymc as pm\nwith pm.Model() as model:\n    pass\n"


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text(MODEL_SRC, encoding="utf-8")
    return models_dir


def _row(rank, elpd_diff, dse, *, untrusted=False):
    return {
        "rank": rank,
        "elpd_loo": -10.0 - elpd_diff,
        "elpd_diff": elpd_diff,
        "dse": dse,
        "dse_clustered": dse,
        "weight": 0.0,
        "loo_unreliable": untrusted,
    }


def _stub_comparison(monkeypatch, rows):
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: rows)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)


def _ledger(tmp_path):
    return HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)


# ── Pruning and the cap: the same rule for every model ─────────────────


def test_a_starting_model_clearly_behind_is_pruned_and_recorded(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b", "discovered"])
    _stub_comparison(
        monkeypatch,
        {
            "discovered": _row(0, 0.0, 0.0),
            "seed_a": _row(1, 1.0, 2.0),  # tied: within 2 clustered SE
            "seed_b": _row(2, 50.0, 5.0),  # clearly behind
        },
    )
    ledger = _ledger(tmp_path)

    pruned = _prune_losers(
        models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={},
        ledger=ledger, ledger_context="experiment1 end of experiment",
    )

    assert pruned == ["seed_b"]
    assert (models_dir / "pruned" / "seed_b.py").exists()
    assert (models_dir / "seed_a.py").exists()
    (entry,) = ledger.entries()
    assert (entry.name, entry.outcome) == ("seed_b", "pruned")
    assert entry.detail.startswith("50.0 nats behind discovered")
    assert entry.hypothesis == "mechanism seed_b"


def test_a_tied_starting_model_is_kept(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "discovered"])
    _stub_comparison(
        monkeypatch, {"discovered": _row(0, 0.0, 0.0), "seed_a": _row(1, 3.0, 2.0)}
    )
    assert _prune_losers(models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={}) == []
    assert (models_dir / "seed_a.py").exists()


def test_the_cap_can_retire_a_starting_model(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "m1", "m2"])
    _stub_comparison(
        monkeypatch,
        {"m1": _row(0, 0.0, 0.0), "m2": _row(1, 1.0, 2.0), "seed_a": _row(2, 2.0, 2.0)},
    )
    ledger = _ledger(tmp_path)

    retired = _cap_live_set(
        models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={}, cap=2,
        ledger=ledger, ledger_context="experiment1 end of experiment",
    )

    assert retired == ["seed_a"]
    assert (models_dir / "pruned" / "seed_a.py").exists()
    assert [(e.name, e.outcome) for e in ledger.entries()] == [("seed_a", "pruned")]


# ── The edge: pruning never empties the set nor loses the best trusted model ──


def test_pruning_keeps_the_best_trusted_model_even_when_everything_else_loses(
    tmp_path, monkeypatch
):
    # Every other model is clearly behind or untrusted: the best trusted model
    # (a starting model here) is the one survivor with a trusted fit.
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b", "flaky", "loser"])
    _stub_comparison(
        monkeypatch,
        {
            "seed_a": _row(0, 0.0, 0.0),
            "seed_b": _row(1, 40.0, 5.0),
            "loser": _row(2, 60.0, 5.0),
            "flaky": _row(3, 70.0, 5.0, untrusted=True),
        },
    )
    pruned = _prune_losers(models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={})
    assert sorted(pruned) == ["loser", "seed_b"]
    # The untrusted model is never pruned (its margin cannot be trusted).
    assert model_zoo._manifest_names(models_dir) == ["seed_a", "flaky"]


def test_the_cap_never_retires_the_best_trusted_model(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "u1", "u2"])
    _stub_comparison(
        monkeypatch,
        {
            "u1": _row(0, 0.0, 0.0, untrusted=True),
            "seed_a": _row(1, 5.0, 2.0),
            "u2": _row(2, 9.0, 2.0, untrusted=True),
        },
    )
    retired = _cap_live_set(
        models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={}, cap=1
    )
    assert sorted(retired) == ["u1", "u2"]
    assert model_zoo._manifest_names(models_dir) == ["seed_a"]


def test_a_cap_below_one_is_refused(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "m1"])
    _stub_comparison(monkeypatch, {"seed_a": _row(0, 0.0, 0.0), "m1": _row(1, 1.0, 1.0)})
    with pytest.raises(ValueError, match="at least one model"):
        _cap_live_set(models_dir, tmp_path / "r.csv", cache_dir=None, fit_kwargs={}, cap=0)


# ── A starting model that cannot be fitted is dropped and recorded ─────


def test_a_starting_model_with_a_nonfinite_logp_is_dropped(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b"])
    monkeypatch.setattr(
        model_zoo,
        "model_logp_is_finite",
        lambda name, *a, **k: (name != "seed_b", "logp is nan"),
    )
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    ledger = _ledger(tmp_path)

    _drop_unfittable_models(
        models_dir, tmp_path / "r.csv", ledger=ledger, ledger_context="experiment1",
        starting_models={"seed_a", "seed_b"},
    )

    assert model_zoo._manifest_names(models_dir) == ["seed_a"]
    assert [(e.name, e.outcome) for e in ledger.entries()] == [("seed_b", "dropped")]


def test_a_starting_model_whose_fit_fails_or_elpd_is_nonfinite_is_dropped(
    tmp_path, monkeypatch
):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b", "seed_c"])
    monkeypatch.setattr(
        model_zoo,
        "fit_models_to_cache",
        lambda names, *a, **k: {"seed_b": "SamplingError: bad initial energy"},
    )
    monkeypatch.setattr(
        model_zoo,
        "log_likelihood",
        lambda name, *a, **k: math.nan if name == "seed_c" else -100.0,
    )
    ledger = _ledger(tmp_path)

    _drop_nonfinite_elpd_models(models_dir, tmp_path / "r.csv", ledger=ledger)

    assert model_zoo._manifest_names(models_dir) == ["seed_a"]
    assert sorted((e.name, e.outcome) for e in ledger.entries()) == [
        ("seed_b", "dropped"),
        ("seed_c", "dropped"),
    ]


def test_an_infrastructure_failure_on_a_starting_model_still_raises(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b"])
    monkeypatch.setattr(
        model_zoo, "fit_models_to_cache", lambda names, *a, **k: {}
    )

    def broken(name, *a, **k):
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(model_zoo, "log_likelihood", broken)
    with pytest.raises(OSError, match="disk quota"):
        _drop_nonfinite_elpd_models(models_dir, tmp_path / "r.csv")
    assert model_zoo._manifest_names(models_dir) == ["seed_a", "seed_b"]


def test_a_starting_model_that_breaks_the_data_contract_still_raises(tmp_path, monkeypatch):
    """Not a fitting failure: a starting model scored on something other than
    the responses is a broken project asset (and would make the fitted
    starting-model baseline meaningless), so it stops the cell."""
    models_dir = _models_dir(tmp_path, ["seed_a", "candidate"])
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        model_zoo, "model_contract_violation", lambda name, *a, **k: "observed data flipped"
    )
    with pytest.raises(RuntimeError, match="Starting model 'seed_a' breaks the data contract"):
        _drop_unfittable_models(
            models_dir, tmp_path / "r.csv", starting_models={"seed_a"}
        )


# ── The whole loop: a starting model can lose and leave ────────────────


def test_the_loop_prunes_a_starting_model_and_the_menu_offers_it(tmp_path, monkeypatch):
    # carried_c sits 30 nats behind model_a at dse 5 (see _patch_scoring).
    seed_dir = write_seed_models(tmp_path, names=("model_a", "model_b", "carried_c"))
    _patch_scoring(monkeypatch)
    run_root = tmp_path / "run"
    results_dir = run_root / "experiment1" / "model_loop"

    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=0,
        enable_critique=False,
        starting_models={"model_a", "model_b", "carried_c"},
        ledger_context="experiment1",
    )

    assert _manifest_names(results_dir / "models") == ["model_a", "model_b"]
    pruned_file = results_dir / "models" / "pruned" / "carried_c.py"
    assert pruned_file.exists()
    rows = _ledger_rows(results_dir / LEDGER_FILENAME)
    assert [(r["name"], r["outcome"]) for r in rows] == [("carried_c", "pruned")]

    # In the next experiment's refinement menu, with its source.
    next_models = _models_dir(run_root / "experiment2" / "model_loop", ["model_a", "model_b"])
    ledger = HypothesisLedger.create(
        tmp_path / "exp2_ledger.jsonl", inherit_from=results_dir / LEDGER_FILENAME
    )
    candidate_dir = tmp_path / "candidate_2"
    candidate_dir.mkdir()
    menu = _write_refinement_menu(candidate_dir, next_models, {}, ledger, incumbent="model_a")
    pruned_part = menu.split("## Pruned", 1)[1]
    assert "### carried_c — pruned (experiment1 end of experiment)" in pruned_part
    assert f"**Source:** `{pruned_file}`" in pruned_part


def test_starting_models_default_to_the_seeded_set_and_are_still_prunable(
    tmp_path, monkeypatch
):
    seed_dir = write_seed_models(tmp_path, names=("model_a", "model_b", "carried_c"))
    _patch_scoring(monkeypatch)
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=1,
        candidate_count=0,
        enable_critique=False,
    )

    assert _manifest_names(results_dir / "models") == ["model_a", "model_b"]


# ── Export, carry-forward and the registry ─────────────────────────────


def _finished_experiment(tmp_path, *, cognitive, zoo, pruned):
    exp_dir = tmp_path / "experiment1"
    cog_dir = _models_dir(exp_dir, cognitive)
    cog_dir.rename(exp_dir / "cognitive_models")
    loop_dir = exp_dir / "model_loop"
    zoo_dir = _models_dir(loop_dir, zoo)
    (zoo_dir / "pruned").mkdir()
    for name in pruned:
        (zoo_dir / "pruned" / f"{name}.py").write_text(MODEL_SRC, encoding="utf-8")
    (loop_dir / "model_posterior.json").write_text(json.dumps({"best_model": zoo[0]}))
    return exp_dir, loop_dir


def test_a_pruned_starting_model_leaves_the_carried_set_and_the_design_prior(tmp_path):
    exp1, loop = _finished_experiment(
        tmp_path,
        cognitive=["seed_a", "seed_b", "seed_c"],
        zoo=["winner", "seed_a"],
        pruned=["seed_b"],  # seed_c was dropped as unfittable: in neither
    )

    _export_inner_loop_models(exp1, loop, best_model="winner")
    update_registry_from_interpretation(exp1)
    exp2 = tmp_path / "experiment2"
    assert carry_forward_cognitive_models(exp1, exp2)

    carried = yaml.safe_load((exp2 / "cognitive_models" / "models_manifest.yaml").read_text())
    assert [m["name"] for m in carried["models"]] == ["seed_a", "winner"]
    for gone in ("seed_b", "seed_c"):
        assert not (exp2 / "cognitive_models" / f"{gone}.py").exists()
    registry = yaml.safe_load((exp1 / "model_registry.yaml").read_text())
    assert registry["theories"] == pytest.approx({"seed_a": 0.5, "winner": 0.5})


# ── The run's record: which models it started from, and that they were prunable ──


def test_the_run_records_its_starting_models_as_prunable(tmp_path):
    exp1 = tmp_path / "run" / "experiment1"
    _models_dir(exp1, ["falk_konold_dp", "local_representativeness"]).rename(
        exp1 / "cognitive_models"
    )

    names = run_starting_models(exp1, "subjective_randomness")

    assert names == {"falk_konold_dp", "local_representativeness"}
    record = json.loads((exp1.parent / STARTING_MODELS_FILENAME).read_text())
    assert record == {
        "starting_models": ["falk_konold_dp", "local_representativeness"],
        "starting_models_prunable": True,
    }
    assert starting_models_prunable(exp1.parent) is True


def test_a_run_that_started_with_protected_starting_models_cannot_continue(tmp_path):
    """A run recorded by the earlier code (a bare list) started with its
    starting models protected; continuing it would mix the two conditions."""
    exp2 = tmp_path / "run" / "experiment2"
    exp2.mkdir(parents=True)
    (exp2.parent / STARTING_MODELS_FILENAME).write_text('["falk_konold_dp"]')

    with pytest.raises(ValueError, match="protected"):
        run_starting_models(exp2, "subjective_randomness")
    assert starting_models_prunable(exp2.parent) is False


def test_a_run_without_a_record_has_no_prunable_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match=STARTING_MODELS_FILENAME):
        starting_models_prunable(tmp_path)


def test_the_wrapper_passes_the_starting_models_as_names_only(tmp_path, monkeypatch):
    exp_dir = tmp_path / "run" / "experiment2"
    _models_dir(exp_dir, ["falk_konold_dp", "carried"]).rename(exp_dir / "cognitive_models")
    (exp_dir.parent / STARTING_MODELS_FILENAME).write_text(
        json.dumps({"starting_models": ["falk_konold_dp", "motif_stack"],
                    "starting_models_prunable": True})
    )
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **kwargs):
        captured.update(kwargs)
        return {"best_model": "carried"}

    monkeypatch.setattr(mlr, "_pooled_response_rows", lambda e: [{"chose_left": "1"}])
    monkeypatch.setattr(mlr, "write_responses_csv", lambda rows, out: out)
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop", fake_inner_loop
    )
    monkeypatch.setattr(mlr, "_export_inner_loop_models", lambda e, l, *, best_model: e)

    mlr.run_inner_model_loop_programmatic(
        exp_dir, max_iterations=0, candidate_count=0, project_id="subjective_randomness"
    )

    assert captured["starting_models"] == {"falk_konold_dp", "motif_stack"}
    assert "protected_names" not in captured


# ── The fitted starting-model baseline still covers every starting model ──


class _FakeFit:
    def loo_diagnostics(self):
        return SimpleNamespace(elpd_loo=-50.0, unreliable=False)

    def convergence_problems(self):
        return []


def test_the_fitted_baseline_fits_a_pruned_starting_model_from_its_pruned_file(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    seeded = _models_dir(run_root / "experiment1" / "model_loop", ["seed_a"])
    (seeded / "pruned").mkdir()
    (seeded / "pruned" / "seed_b.py").write_text(MODEL_SRC, encoding="utf-8")
    assert holdout_eval.seeded_models_dir(run_root) == seeded
    fitted_from = {}

    def fake_fit(name, models_dir, responses_path, **kwargs):
        fitted_from[name] = models_dir
        return _FakeFit()

    monkeypatch.setattr(holdout_eval, "fit_model", fake_fit)
    monkeypatch.setattr(
        holdout_eval, "_eval_prediction", lambda fitted, rows, **k: np.array([0.2, 0.6, 0.7])
    )

    holdout_eval._require_seeded(["seed_a", "seed_b"], seeded)
    baseline = holdout_eval._fitted_seed_baseline(
        ["seed_a", "seed_b"], seeded, tmp_path / "r.csv", [{}, {}, {}],
        np.array([0.1, 0.5, 0.9]), participant_ids=None, cache_dir=None, fit_kwargs={},
    )

    assert set(baseline["per_model"]) == {"seed_a", "seed_b"}
    assert fitted_from == {"seed_a": seeded, "seed_b": seeded / "pruned"}


def test_the_baseline_still_refuses_a_starting_model_the_run_never_had(tmp_path):
    seeded = _models_dir(tmp_path / "run" / "experiment1" / "model_loop", ["seed_a"])
    with pytest.raises(FileNotFoundError, match="seed_b"):
        holdout_eval._require_seeded(["seed_a", "seed_b"], seeded)
