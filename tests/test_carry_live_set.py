"""The loop remembers what it learned across experiments.

Before this change only the single exported winner crossed an experiment
boundary. Every other model the inner loop admitted — including rivals
statistically tied with the winner — stayed behind in ``model_loop/models/``,
the next experiment's candidate agents re-proposed it (15 % of candidate slots
in the iteration-2 recovery sweep re-proposed a name already tried in the same
cell), and the registry handed the next design stacking weights over models
that were not in its set (15 of 40 designs had a degenerate prior and 32
zero-EIG stimuli).

After the inner loop, ``cognitive_models/`` is now the **live set**: the
protected seeds plus every zoo survivor. A carried model the loop pruned leaves
it, the ledger of attempted hypotheses travels with it, and the design prior is
uniform over exactly the models the next design scores.
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.outer_loop.orchestrator import (
    _export_inner_loop_models,
    carry_forward_cognitive_models,
    update_registry_from_interpretation,
)

MODEL_SRC = "import pymc as pm\nwith pm.Model() as model:\n    pass\n"
LEDGER_TEXT = (
    json.dumps(
        {
            "name": "loser",
            "outcome": "pruned",
            "detail": "40.0 nats behind winner (8.0× dse)",
            "hypothesis": "People count runs.",
            "context": "experiment1 round 0",
        }
    )
    + "\n"
)


def _write_manifest(models_dir, names):
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _finished_experiment(tmp_path, *, cognitive, zoo, pruned, best="winner"):
    """An experiment whose inner loop just finished.

    ``cognitive`` is the set the experiment started from (seeds + carried
    models); ``zoo`` the surviving inner-loop set; ``pruned`` the models the
    loop moved to ``models/pruned/`` — a carried model can be among them.
    """
    exp_dir = tmp_path / "experiment1"
    cog_dir = exp_dir / "cognitive_models"
    cog_dir.mkdir(parents=True)
    _write_manifest(cog_dir, cognitive)
    for name in cognitive:
        (cog_dir / f"{name}.py").write_text(MODEL_SRC, encoding="utf-8")

    loop_dir = exp_dir / "model_loop"
    zoo_dir = loop_dir / "models"
    (zoo_dir / "pruned").mkdir(parents=True)
    _write_manifest(zoo_dir, zoo)
    for name in zoo:
        (zoo_dir / f"{name}.py").write_text(MODEL_SRC, encoding="utf-8")
    for name in pruned:
        (zoo_dir / "pruned" / f"{name}.py").write_text(MODEL_SRC, encoding="utf-8")
    (loop_dir / LEDGER_FILENAME).write_text(LEDGER_TEXT, encoding="utf-8")
    (loop_dir / "model_posterior.json").write_text(
        json.dumps(
            {
                "best_model": best,
                "posteriors": {n: (1.0 if n == best else 0.0) for n in zoo},
                "elpd_loo": {n: -100.0 for n in zoo},
                "comparison": {
                    n: {"rank": i, "elpd_loo": -100.0, "elpd_diff": float(i),
                        "dse": 1.0, "weight": 1.0 if n == best else 0.0,
                        "loo_unreliable": False}
                    for i, n in enumerate([best] + [m for m in zoo if m != best])
                },
            }
        ),
        encoding="utf-8",
    )
    return exp_dir, loop_dir


def _manifest_names(exp_dir):
    manifest = yaml.safe_load(
        (exp_dir / "cognitive_models" / "models_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    return [m["name"] for m in manifest["models"]]


def test_next_experiment_starts_from_the_live_set_with_its_ledger_and_a_uniform_prior(
    tmp_path,
):
    exp1, loop = _finished_experiment(
        tmp_path,
        cognitive=["seed_a", "seed_b", "carried_old"],
        zoo=["seed_a", "seed_b", "winner", "rival"],
        pruned=["carried_old", "loser"],
    )

    _export_inner_loop_models(
        exp1, loop, best_model="winner", protected_names={"seed_a", "seed_b"}
    )
    update_registry_from_interpretation(exp1)
    exp2 = tmp_path / "experiment2"
    assert carry_forward_cognitive_models(exp1, exp2)

    # The live set: seeds first (their original order), then every zoo survivor
    # in zoo order; the carried model the loop pruned is gone, file and entry.
    assert _manifest_names(exp2) == ["seed_a", "seed_b", "winner", "rival"]
    for exp in (exp1, exp2):
        assert not (exp / "cognitive_models" / "carried_old.py").exists()
        assert (exp / "cognitive_models" / "rival.py").exists()
    # The ledger travels with the model set.
    assert (exp2 / "cognitive_models" / LEDGER_FILENAME).read_text(
        encoding="utf-8"
    ) == LEDGER_TEXT
    # The design prior is uniform over exactly the carried set.
    registry = yaml.safe_load((exp1 / "model_registry.yaml").read_text(encoding="utf-8"))
    assert registry["theories"] == pytest.approx(
        {"seed_a": 0.25, "seed_b": 0.25, "winner": 0.25, "rival": 0.25}
    )
    assert registry["reserved_for_new"] == 0.0


def test_export_keeps_a_protected_seed_the_loop_dropped_as_unfittable(tmp_path):
    # seed_b never made it into the zoo (dropped as unfittable on this data);
    # it is a protected baseline and stays in the carried set regardless.
    exp1, loop = _finished_experiment(
        tmp_path,
        cognitive=["seed_a", "seed_b"],
        zoo=["seed_a", "winner"],
        pruned=[],
    )
    _export_inner_loop_models(
        exp1, loop, best_model="winner", protected_names={"seed_a", "seed_b"}
    )
    assert _manifest_names(exp1) == ["seed_a", "seed_b", "winner"]
    assert (exp1 / "cognitive_models" / "seed_b.py").exists()


def test_export_is_idempotent(tmp_path):
    exp1, loop = _finished_experiment(
        tmp_path,
        cognitive=["seed_a", "carried_old"],
        zoo=["seed_a", "winner"],
        pruned=["carried_old"],
    )
    for _ in range(2):
        _export_inner_loop_models(
            exp1, loop, best_model="winner", protected_names={"seed_a"}
        )
    assert _manifest_names(exp1) == ["seed_a", "winner"]
    assert (exp1 / "cognitive_models" / LEDGER_FILENAME).exists()


def test_export_refuses_a_best_model_outside_the_zoo(tmp_path):
    exp1, loop = _finished_experiment(
        tmp_path, cognitive=["seed_a"], zoo=["seed_a", "winner"], pruned=[]
    )
    with pytest.raises(ValueError, match="not in the inner-loop zoo"):
        _export_inner_loop_models(
            exp1, loop, best_model="ghost", protected_names={"seed_a"}
        )


def test_carry_forward_without_a_ledger_copies_the_model_set_only(tmp_path):
    exp1, loop = _finished_experiment(
        tmp_path, cognitive=["seed_a"], zoo=["seed_a", "winner"], pruned=[]
    )
    (loop / LEDGER_FILENAME).unlink()
    _export_inner_loop_models(exp1, loop, best_model="winner", protected_names={"seed_a"})
    exp2 = tmp_path / "experiment2"
    assert carry_forward_cognitive_models(exp1, exp2)
    assert _manifest_names(exp2) == ["seed_a", "winner"]
    assert not (exp2 / "cognitive_models" / LEDGER_FILENAME).exists()


def test_wrapper_protects_only_the_project_seeds_and_labels_the_ledger(
    tmp_path, monkeypatch
):
    """The outer loop tells the inner loop which models are the project's
    seeds (never pruned, always carried); a model carried from an earlier
    experiment is not protected, and the ledger is labelled by experiment."""
    from src.pipelines.outer_loop import orchestrator as orch

    exp_dir = tmp_path / "holdout" / "some_gt" / "experiment2"
    cog_dir = exp_dir / "cognitive_models"
    cog_dir.mkdir(parents=True)
    # Two real project seeds plus a model carried from experiment 1.
    _write_manifest(cog_dir, ["falk_konold_dp", "motif_stack", "carried_from_exp1"])
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **inner_kwargs):
        captured.update(inner_kwargs)
        return {"best_model": "carried_from_exp1"}

    monkeypatch.setattr(orch, "_pooled_response_rows", lambda e: [{"chose_left": "1"}])
    monkeypatch.setattr(orch, "_load_project_featurizer", lambda project_dir: None)
    monkeypatch.setattr(orch, "_write_feature_csv", lambda rows, fz, out: out)
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )

    def fake_export(e, l, *, best_model, protected_names):
        captured["export_protected"] = set(protected_names)
        return e

    monkeypatch.setattr(orch, "_export_inner_loop_models", fake_export)

    orch.run_inner_model_loop_programmatic(
        exp_dir, max_iterations=0, candidate_count=0, project_id="subjective_randomness"
    )

    assert captured["protected_names"] == {"falk_konold_dp", "motif_stack"}
    assert captured["export_protected"] == {"falk_konold_dp", "motif_stack"}
    assert captured["ledger_context"] == "experiment2"
