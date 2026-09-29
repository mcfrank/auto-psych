"""Pruning happens once, at the end of each experiment, and the live set is capped.

Pruning used to run after every round, against the max of a growing zoo, and a
model lost there left for good. It now runs once per experiment, after the
last round (user decision 2026-09-26). After it, the live set carried to the
next experiment is capped at MAX_LIVE_MODELS: beyond it, the models that
cannot be trusted (unreliable PSIS-LOO or a non-converged fit) retire first,
then the lowest by ELPD. Seeds are never retired.
"""

from __future__ import annotations

import yaml

from src.pipelines.inner_loop import model_zoo, pymc_orchestrator
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME, HypothesisLedger
from src.pipelines.inner_loop.model_zoo import _cap_live_set
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from tests.inner_loop_fixtures import canned_posterior, write_responses, write_seed_models
from tests.test_pymc_inner_loop_history import _patch_candidates, _patch_scoring


def test_pruning_runs_once_after_the_last_round(tmp_path, monkeypatch):
    _patch_scoring(
        monkeypatch,
        [
            canned_posterior("model_a", ["model_b"]),
            canned_posterior("model_a", ["model_b", "iter0_candidate0"]),
            canned_posterior("model_a", ["model_b", "iter0_candidate0", "iter1_candidate0"]),
        ],
    )
    _patch_candidates(monkeypatch)
    calls = []
    monkeypatch.setattr(
        pymc_orchestrator, "_prune_losers",
        lambda *a, **k: calls.append(("prune", k["ledger_context"])) or [],
    )
    monkeypatch.setattr(
        pymc_orchestrator, "_cap_live_set",
        lambda *a, **k: calls.append(("cap", k["ledger_context"])) or [],
    )
    run_pymc_inner_loop(
        write_responses(tmp_path),
        tmp_path / "results",
        seed_models_dir=write_seed_models(tmp_path),
        max_iterations=2,
        candidate_count=1,
        ledger_context="experiment1",
    )
    assert calls == [
        ("prune", "experiment1 end of experiment"),
        ("cap", "experiment1 end of experiment"),
    ]


def _zoo(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": n, "rationale": f"H {n}"} for n in names]}),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text("# model\n", encoding="utf-8")
    return models_dir


def _row(rank, **flags):
    return {"rank": rank, "elpd_loo": -100.0 - 10 * rank, "elpd_diff": 10.0 * rank,
            "dse": 3.0, **flags}


def test_the_cap_retires_untrusted_models_first_then_the_lowest_elpd(tmp_path, monkeypatch):
    models_dir = _zoo(tmp_path, ["seed", "a1", "a2", "a3", "a4"])
    comparison = {
        "seed": _row(0), "a1": _row(1), "a2": _row(2),
        "a3": _row(3, not_converged=True), "a4": _row(4),
    }
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: comparison)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    ledger = HypothesisLedger.create(tmp_path / LEDGER_FILENAME, inherit_from=None)

    retired = _cap_live_set(
        models_dir, tmp_path / "r.csv", protected={"seed"}, cache_dir=None,
        fit_kwargs={}, cap=3, ledger=ledger, ledger_context="experiment1 end of experiment",
    )

    assert retired == ["a3", "a4"]
    assert model_zoo._manifest_names(models_dir) == ["seed", "a1", "a2"]
    assert (models_dir / "pruned" / "a4.py").exists()
    details = {e.name: e.detail for e in ledger.entries()}
    assert "live set" in details["a4"] and "cannot be trusted" in details["a3"]


def test_seeds_are_never_retired_by_the_cap(tmp_path, monkeypatch):
    models_dir = _zoo(tmp_path, ["s1", "s2", "a1"])
    comparison = {"a1": _row(0), "s1": _row(1), "s2": _row(2)}
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: comparison)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    retired = _cap_live_set(
        models_dir, tmp_path / "r.csv", protected={"s1", "s2"}, cache_dir=None,
        fit_kwargs={}, cap=2, ledger=None, ledger_context="",
    )
    assert retired == ["a1"]
    assert model_zoo._manifest_names(models_dir) == ["s1", "s2"]


def test_a_set_within_the_cap_is_untouched(tmp_path, monkeypatch):
    models_dir = _zoo(tmp_path, ["seed", "a1"])
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: 1 / 0)
    assert _cap_live_set(
        models_dir, tmp_path / "r.csv", protected={"seed"}, cache_dir=None,
        fit_kwargs={}, cap=8, ledger=None, ledger_context="",
    ) == []
