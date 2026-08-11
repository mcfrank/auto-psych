"""Pruning: agent models that obviously lose leave the zoo.

Without pruning the model set only grows: every scoring pass re-ranks every
model ever admitted, every InferenceData stays resident in the fit cache, and
existing_hypotheses.md drags dead hypotheses into every candidate prompt. A
model is pruned only when BOTH hold on the current data: it is statistically
distinguishable from the best (``elpd_diff > multiplier·dse``) AND its stacking
weight is negligible (< floor). The seeded set is never pruned — those are the
baselines the run reports against. Pruned files move to ``models/pruned/`` (an
audit trail, not a deletion).
"""

from __future__ import annotations

import json

import yaml
import pytest

from src.pipelines.inner_loop import pymc_orchestrator
from src.pipelines.inner_loop.pymc_orchestrator import _export, _prune_losers


def _models_dir(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]}
        ),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text("# model\n", encoding="utf-8")
        (models_dir / f"{n}.hypothesis.md").write_text(f"H {n}\n", encoding="utf-8")
    return models_dir


def _stub_comparison(monkeypatch, rows):
    monkeypatch.setattr(
        pymc_orchestrator, "compare_table", lambda *a, **k: rows
    )
    evicted = []
    monkeypatch.setattr(
        pymc_orchestrator, "evict_fit_cache", lambda name: evicted.append(name)
    )
    return evicted


def _row(rank, elpd_diff, dse, weight):
    return {
        "rank": rank,
        "elpd_loo": -10.0 - elpd_diff,
        "elpd_diff": elpd_diff,
        "dse": dse,
        "weight": weight,
        "loo_unreliable": False,
    }


def test_prunes_distinguishable_negligible_agent_model(tmp_path, monkeypatch, capsys):
    models_dir = _models_dir(tmp_path, ["seed_a", "dead_end"])
    evicted = _stub_comparison(
        monkeypatch,
        {
            "seed_a": _row(0, 0.0, 0.0, 0.995),
            "dead_end": _row(1, 12.0, 2.0, 0.005),
        },
    )
    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
    )
    assert pruned == ["dead_end"]
    assert not (models_dir / "dead_end.py").exists()
    assert (models_dir / "pruned" / "dead_end.py").exists()
    assert (models_dir / "pruned" / "dead_end.hypothesis.md").exists()
    manifest = yaml.safe_load(
        (models_dir / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    assert [m["name"] for m in manifest["models"]] == ["seed_a"]
    assert evicted == ["dead_end"]
    assert "dead_end" in capsys.readouterr().out


def test_protected_models_are_never_pruned(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "seed_b"])
    _stub_comparison(
        monkeypatch,
        {
            "seed_a": _row(0, 0.0, 0.0, 0.999),
            "seed_b": _row(1, 50.0, 2.0, 0.001),  # loses badly, but protected
        },
    )
    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a", "seed_b"},
        cache_dir=None,
        fit_kwargs=None,
    )
    assert pruned == []
    assert (models_dir / "seed_b.py").exists()


def test_indistinguishable_or_weighted_models_stay(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "near_tie", "still_weighted"])
    _stub_comparison(
        monkeypatch,
        {
            "seed_a": _row(0, 0.0, 0.0, 0.5),
            # Within 2*dse of the best: statistically indistinguishable.
            "near_tie": _row(1, 1.5, 1.0, 0.005),
            # Distinguishable but still carries stacking weight above the floor.
            "still_weighted": _row(2, 10.0, 2.0, 0.05),
        },
    )
    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
    )
    assert pruned == []


def test_unreliable_loser_is_not_pruned(tmp_path, monkeypatch, capsys):
    """A model whose own LOO is unreliable may not be pruned on that estimate."""
    models_dir = _models_dir(tmp_path, ["seed_a", "dead_end"])
    rows = {
        "seed_a": _row(0, 0.0, 0.0, 0.995),
        "dead_end": _row(1, 12.0, 2.0, 0.005),
    }
    rows["dead_end"]["loo_unreliable"] = True
    evicted = _stub_comparison(monkeypatch, rows)

    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
    )

    assert pruned == []
    assert (models_dir / "dead_end.py").exists()
    assert evicted == []
    assert "unreliable" in capsys.readouterr().err.lower()


def test_unreliable_bystander_does_not_block_pruning_reliable_losers(
    tmp_path, monkeypatch, capsys
):
    """One flaky candidate must not globally disable pruning.

    Agent-written models trip Pareto-k warnings routinely; if any single
    unreliable row switched pruning off wholesale, the active set could only
    grow for the rest of the run. Only the unreliable model itself is spared —
    a clear loser with a sound LOO row still leaves the zoo.
    """
    models_dir = _models_dir(tmp_path, ["seed_a", "flaky", "dead_end"])
    rows = {
        "seed_a": _row(0, 0.0, 0.0, 0.90),
        "flaky": _row(1, 1.0, 1.0, 0.09),
        "dead_end": _row(2, 12.0, 2.0, 0.005),
    }
    rows["flaky"]["loo_unreliable"] = True
    evicted = _stub_comparison(monkeypatch, rows)

    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
    )

    assert pruned == ["dead_end"]
    assert (models_dir / "flaky.py").exists()
    assert not (models_dir / "dead_end.py").exists()
    assert evicted == ["dead_end"]
    err = capsys.readouterr().err.lower()
    assert "flaky" in err and "unreliable" in err


def test_empty_comparison_prunes_nothing(tmp_path, monkeypatch):
    """No comparison rows (e.g. a stubbed or degenerate compare) = no pruning."""
    models_dir = _models_dir(tmp_path, ["seed_a", "dead_end"])
    _stub_comparison(monkeypatch, {})

    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected=set(),
        cache_dir=None,
        fit_kwargs=None,
    )

    assert pruned == []
    assert (models_dir / "dead_end.py").exists()


def test_unreliable_baseline_blocks_all_pruning(tmp_path, monkeypatch, capsys):
    """Every elpd_diff is measured against the rank-0 model; if ITS LOO is
    unreliable, no difference is trustworthy and nothing may be pruned."""
    models_dir = _models_dir(tmp_path, ["seed_a", "dead_end"])
    rows = {
        "seed_a": _row(0, 0.0, 0.0, 0.995),
        "dead_end": _row(1, 12.0, 2.0, 0.005),
    }
    rows["seed_a"]["loo_unreliable"] = True
    evicted = _stub_comparison(monkeypatch, rows)

    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
    )

    assert pruned == []
    assert (models_dir / "dead_end.py").exists()
    assert evicted == []
    assert "unreliable" in capsys.readouterr().err.lower()


def test_unreliable_selected_model_prevents_result_export(tmp_path):
    models_dir = _models_dir(tmp_path, ["selected", "runner_up"])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    posterior = {
        "posteriors": {"selected": 0.8, "runner_up": 0.2},
        "elpd_loo": {"selected": -10.0, "runner_up": -11.0},
        "n_trials": 20,
    }
    comparison = {
        "selected": {**_row(0, 0.0, 0.0, 0.8), "loo_unreliable": True},
        "runner_up": _row(1, 1.0, 1.0, 0.2),
    }

    with pytest.raises(RuntimeError, match="selected.*LOO.*unreliable"):
        _export(results_dir, models_dir, posterior, comparison)

    # The refusal blocks best_model.py, not the record of WHY it refused:
    # model_posterior.json (posterior + comparison, unreliable flag included)
    # must land on disk so a refused run can be diagnosed after the fact.
    assert not (results_dir / "best_model.py").exists()
    payload = json.loads(
        (results_dir / "model_posterior.json").read_text(encoding="utf-8")
    )
    assert payload["posteriors"] == posterior["posteriors"]
    assert payload["comparison"]["selected"]["loo_unreliable"] is True


def test_zero_multiplier_disables_pruning(tmp_path, monkeypatch):
    models_dir = _models_dir(tmp_path, ["seed_a", "dead_end"])

    def tripwire(*a, **k):
        raise AssertionError("compare_table must not run when pruning is disabled")

    monkeypatch.setattr(pymc_orchestrator, "compare_table", tripwire)
    pruned = _prune_losers(
        models_dir,
        tmp_path / "responses.csv",
        protected={"seed_a"},
        cache_dir=None,
        fit_kwargs=None,
        dse_multiplier=0.0,
    )
    assert pruned == []
