"""The run's starting models are recorded once; retired names stay taken.

- Second audit B10: the starting models were recomputed every experiment as
  the project's seed manifest intersected with the carried set, so a
  candidate that chose a seed's name -- the held-out ground truth's, in a
  holdout cell -- was carried from then on as a starting model. The run's
  starting models are now recorded when experiment 1's model loop first runs
  (``<run>/starting_models.json``). Since 2026-09-28 they are not protected
  from pruning (tests/test_starting_models_prunable.py); their names stay
  reserved.
- A candidate may not take a starting model's name (first audit D3 too):
  nor the name of a model pruned, retired or dropped earlier in the run, which
  made the ledger, the refinement menu and the evaluation (one file per name)
  conflate two models. It is renamed like any other clash, and the admitted
  ledger entry records it.
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME, HypothesisLedger
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.pipelines.outer_loop import model_loop_runner as mlr
from tests.inner_loop_fixtures import write_responses, write_seed_models
from tests.test_pymc_inner_loop_ledger import (
    INHERITED,
    _patch_candidates,
    _patch_scoring,
)

PROJECT = "subjective_randomness"


def _cognitive_models(exp_dir, names):
    cog = exp_dir / "cognitive_models"
    cog.mkdir(parents=True)
    (cog / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]}
        ),
        encoding="utf-8",
    )
    return cog


def _capture_the_loop(monkeypatch):
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **kwargs):
        captured.update(kwargs)
        return {"best_model": "falk_konold_dp"}

    def fake_export(exp_dir, loop_dir, *, best_model):
        return exp_dir

    monkeypatch.setattr(mlr, "_pooled_response_rows", lambda e: [{"chose_left": "1"}])
    monkeypatch.setattr(mlr, "write_responses_csv", lambda rows, out: out)
    monkeypatch.setattr(mlr, "_export_inner_loop_models", fake_export)
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )
    return captured


def _run(exp_dir):
    mlr.run_inner_model_loop_programmatic(
        exp_dir, max_iterations=0, candidate_count=0, project_id=PROJECT
    )


def test_experiment_1_records_the_runs_starting_models(tmp_path, monkeypatch):
    run = tmp_path / "run0" / "motif_stack"
    # A holdout cell: motif_stack (a project seed) is the held-out model.
    _cognitive_models(
        run / "experiment1", ["falk_konold_dp", "local_representativeness"]
    )
    captured = _capture_the_loop(monkeypatch)

    _run(run / "experiment1")

    recorded = json.loads((run / "starting_models.json").read_text(encoding="utf-8"))
    assert recorded == {
        "starting_models": ["falk_konold_dp", "local_representativeness"],
        "starting_models_prunable": True,
    }
    assert captured["starting_models"] == {"falk_konold_dp", "local_representativeness"}


def test_a_candidate_named_after_the_held_out_model_is_not_a_starting_model(
    tmp_path, monkeypatch
):
    run = tmp_path / "run0" / "motif_stack"
    _cognitive_models(
        run / "experiment1", ["falk_konold_dp", "local_representativeness"]
    )
    captured = _capture_the_loop(monkeypatch)
    _run(run / "experiment1")
    # Experiment 1 exported a candidate that called itself motif_stack.
    _cognitive_models(
        run / "experiment2",
        ["falk_konold_dp", "local_representativeness", "motif_stack"],
    )

    _run(run / "experiment2")

    assert captured["starting_models"] == {"falk_konold_dp", "local_representativeness"}


def test_a_resumed_experiment_1_reads_the_record_not_the_exported_set(
    tmp_path, monkeypatch
):
    run = tmp_path / "run0" / "motif_stack"
    (run).mkdir(parents=True)
    (run / "starting_models.json").write_text(
        json.dumps(
            {"starting_models": ["falk_konold_dp"], "starting_models_prunable": True}
        ),
        encoding="utf-8",
    )
    # The export already ran once and carried a candidate named like a seed.
    _cognitive_models(run / "experiment1", ["falk_konold_dp", "motif_stack"])
    captured = _capture_the_loop(monkeypatch)

    _run(run / "experiment1")

    assert captured["starting_models"] == {"falk_konold_dp"}


def test_a_later_experiment_without_a_record_raises(tmp_path, monkeypatch):
    run = tmp_path / "run0" / "motif_stack"
    _cognitive_models(run / "experiment2", ["falk_konold_dp", "motif_stack"])
    _capture_the_loop(monkeypatch)
    with pytest.raises(FileNotFoundError, match="starting_models.json"):
        _run(run / "experiment2")


# ── Names a candidate may not take ───────────────────────────────────────


def test_reserved_names_are_the_starting_models_and_every_retired_model(tmp_path):
    models_dir = tmp_path / "models"
    (models_dir / "pruned").mkdir(parents=True)
    (models_dir / "pruned" / "retired_here.py").write_text(
        "# model\n", encoding="utf-8"
    )
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    for name, outcome in [
        ("pruned_earlier", "pruned"),
        ("unfittable", "dropped"),
        ("never_admitted", "rejected"),
        ("live_one", "admitted"),
    ]:
        model_zoo._record(
            ledger, name=name, outcome=outcome, detail="d", hypothesis="h", context="c"
        )

    reserved = model_zoo.reserved_names(models_dir, ledger, {"seed_a", "held_out_seed"})

    assert reserved == {
        "seed_a",
        "held_out_seed",
        "pruned_earlier",
        "unfittable",
        "retired_here",
    }


def test_the_loop_renames_a_candidate_that_asks_for_a_retired_or_starting_name(
    tmp_path, monkeypatch
):
    seed_dir = write_seed_models(tmp_path)  # model_a, model_b
    (seed_dir / LEDGER_FILENAME).write_text(
        json.dumps(INHERITED) + "\n", encoding="utf-8"
    )
    _patch_scoring(monkeypatch)
    # Round 0 asks for the pruned `old_idea`; round 1 for `gone_seed`, a
    # starting model the run no longer carries.
    _patch_candidates(monkeypatch, {0: "old_idea", 1: "gone_seed"}, [])
    monkeypatch.setattr(
        model_zoo, "_min_prediction_rmse", lambda *a, **k: (None, float("inf"))
    )
    results_dir = tmp_path / "model_loop"

    run_pymc_inner_loop(
        write_responses(tmp_path),
        results_dir,
        seed_models_dir=seed_dir,
        max_iterations=2,
        candidate_count=1,
        enable_critique=False,
        starting_models={"model_a", "model_b", "gone_seed"},
    )

    rows = [
        json.loads(line)
        for line in (results_dir / LEDGER_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    admitted = [row for row in rows if row["outcome"] == "admitted"]
    assert [row["name"] for row in admitted] == ["old_idea_2", "gone_seed_2"]
    assert "asked for the name 'old_idea', which was taken" in admitted[0]["detail"]
    assert "admitted as 'gone_seed_2'" in admitted[1]["detail"]
    live = yaml.safe_load((results_dir / "models" / "models_manifest.yaml").read_text())
    assert "old_idea" not in {entry["name"] for entry in live["models"]}
