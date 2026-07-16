"""Exporting the inner loop's best model into cognitive_models/.

New semantics for the hero run: the export keeps the model's own descriptive
name and its hypothesis as the manifest rationale, and it only copies when the
best model is genuinely new — a seed (or previously exported model) that wins
again is already in the set, so re-exporting it under a second name would split
posterior mass between two identical models in every later experiment. A
fallback-auto-named winner (``iterN_candidateM``) exports under the legacy
stable name ``inner_loop_model`` so the carried manifest never contains zoo
names (which the model-set validator rejects).
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import (
    _export_inner_loop_model,
    _validate_model_loop,
)

MODEL_SRC = "import pymc as pm\nwith pm.Model() as model:\n    pass\n"


def _setup(tmp_path, cognitive_names, zoo_entries):
    """An experiment dir with a cognitive_models set and an inner-loop zoo."""
    exp_dir = tmp_path / "experiment1"
    cog_dir = exp_dir / "cognitive_models"
    cog_dir.mkdir(parents=True)
    (cog_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in cognitive_names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for name in cognitive_names:
        (cog_dir / f"{name}.py").write_text(MODEL_SRC, encoding="utf-8")

    loop_dir = exp_dir / "model_loop"
    zoo_dir = loop_dir / "models"
    zoo_dir.mkdir(parents=True)
    (zoo_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": zoo_entries}, sort_keys=False), encoding="utf-8"
    )
    for entry in zoo_entries:
        (zoo_dir / f"{entry['name']}.py").write_text(MODEL_SRC, encoding="utf-8")
    return exp_dir, loop_dir


def _manifest_names(exp_dir):
    manifest = yaml.safe_load(
        (exp_dir / "cognitive_models" / "models_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    return [m["name"] for m in manifest["models"]]


def test_new_descriptive_winner_is_exported_with_its_hypothesis(tmp_path):
    exp_dir, loop_dir = _setup(
        tmp_path,
        cognitive_names=["seed_a"],
        zoo_entries=[
            {"name": "seed_a", "rationale": "mechanism seed_a"},
            {"name": "recency_weighted_runs", "rationale": "People weight recent runs."},
        ],
    )
    path = _export_inner_loop_model(exp_dir, loop_dir, best_model="recency_weighted_runs")
    assert path == exp_dir / "cognitive_models" / "recency_weighted_runs.py"
    assert path.exists()
    manifest = yaml.safe_load(
        (exp_dir / "cognitive_models" / "models_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    entry = {m["name"]: m["rationale"] for m in manifest["models"]}[
        "recency_weighted_runs"
    ]
    assert "People weight recent runs." in entry


def test_winning_seed_is_not_duplicated(tmp_path):
    exp_dir, loop_dir = _setup(
        tmp_path,
        cognitive_names=["seed_a", "seed_b"],
        zoo_entries=[
            {"name": "seed_a", "rationale": "mechanism seed_a"},
            {"name": "seed_b", "rationale": "mechanism seed_b"},
        ],
    )
    _export_inner_loop_model(exp_dir, loop_dir, best_model="seed_a")
    # No inner_loop_model copy, no duplicate manifest entry.
    assert _manifest_names(exp_dir) == ["seed_a", "seed_b"]
    assert not (exp_dir / "cognitive_models" / "inner_loop_model.py").exists()


def test_fallback_named_winner_exports_as_inner_loop_model(tmp_path):
    exp_dir, loop_dir = _setup(
        tmp_path,
        cognitive_names=["seed_a"],
        zoo_entries=[
            {"name": "seed_a", "rationale": "mechanism seed_a"},
            {"name": "iter0_candidate2", "rationale": "An unnamed hypothesis."},
        ],
    )
    path = _export_inner_loop_model(exp_dir, loop_dir, best_model="iter0_candidate2")
    # Zoo names must never enter the carried manifest (the validator rejects
    # them), so the fallback maps to the legacy stable export name.
    assert path == exp_dir / "cognitive_models" / "inner_loop_model.py"
    assert "inner_loop_model" in _manifest_names(exp_dir)
    assert "iter0_candidate2" not in _manifest_names(exp_dir)


def test_export_missing_zoo_rationale_raises(tmp_path):
    exp_dir, loop_dir = _setup(
        tmp_path,
        cognitive_names=["seed_a"],
        zoo_entries=[{"name": "seed_a", "rationale": "mechanism seed_a"}],
    )
    with pytest.raises(ValueError, match="no_such_model"):
        _export_inner_loop_model(exp_dir, loop_dir, best_model="no_such_model")


def _write_loop_outputs(exp_dir, best):
    loop_dir = exp_dir / "model_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    (loop_dir / "model_posterior.json").write_text(
        json.dumps(
            {
                "posteriors": {best: 0.9, "other": 0.1},
                "comparison": {
                    best: {"weight": 0.8},
                    "other": {"weight": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    (loop_dir / "report.md").write_text("# report\n", encoding="utf-8")


def test_validate_model_loop_accepts_best_in_model_set(tmp_path):
    exp_dir, _ = _setup(
        tmp_path,
        cognitive_names=["seed_a"],
        zoo_entries=[{"name": "seed_a", "rationale": "mechanism seed_a"}],
    )
    _write_loop_outputs(exp_dir, best="seed_a")
    ok, msg = _validate_model_loop(exp_dir)
    assert ok, msg


def test_validate_model_loop_rejects_missing_best(tmp_path):
    exp_dir, _ = _setup(
        tmp_path,
        cognitive_names=["seed_a"],
        zoo_entries=[{"name": "seed_a", "rationale": "mechanism seed_a"}],
    )
    _write_loop_outputs(exp_dir, best="never_exported")
    ok, msg = _validate_model_loop(exp_dir)
    assert not ok
    assert "never_exported" in msg
