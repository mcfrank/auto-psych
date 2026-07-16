"""Programmatic carry-forward of cognitive_models between experiments.

The outer-loop theorist agent was removed for the hero run: new hypotheses now
enter ONLY via the inner loop. The one mechanical job the theorist still did —
copying the previous experiment's cognitive_models/ (model set + the exported
inner-loop best) into the next experiment — is now this deterministic step,
mirroring seed_experiment_models_from_project's contract: True on copy, False
when the destination already has a manifest, loud raise on a broken source.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import (
    carry_forward_cognitive_models,
    seed_experiment_models_from_project,
)


def _prev_experiment_with_export(tmp_path):
    """A completed experiment1: seeded models plus an exported inner-loop best."""
    prev = tmp_path / "experiment1"
    assert seed_experiment_models_from_project(prev, "subjective_randomness")
    models_dir = prev / "cognitive_models"
    manifest_path = models_dir / "models_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    first = manifest["models"][0]["name"]
    shutil.copyfile(models_dir / f"{first}.py", models_dir / "inner_loop_model.py")
    manifest["models"].append(
        {"name": "inner_loop_model", "rationale": "exported inner-loop best"}
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return prev, manifest


def test_carry_forward_copies_every_manifest_model(tmp_path):
    prev, manifest = _prev_experiment_with_export(tmp_path)
    new = tmp_path / "experiment2"

    assert carry_forward_cognitive_models(prev, new)

    copied = yaml.safe_load(
        (new / "cognitive_models" / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    names = [m["name"] for m in copied["models"]]
    assert names == [m["name"] for m in manifest["models"]]
    assert "inner_loop_model" in names
    for name in names:
        assert (new / "cognitive_models" / f"{name}.py").exists()


def test_carry_forward_skips_already_populated_destination(tmp_path):
    prev, _ = _prev_experiment_with_export(tmp_path)
    new = tmp_path / "experiment2"
    assert carry_forward_cognitive_models(prev, new)
    # Resume semantics: a destination that already has a manifest is left alone.
    assert not carry_forward_cognitive_models(prev, new)


def test_carry_forward_missing_previous_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="models_manifest"):
        carry_forward_cognitive_models(
            tmp_path / "experiment1", tmp_path / "experiment2"
        )


def test_carry_forward_empty_previous_manifest_raises(tmp_path):
    prev = tmp_path / "experiment1"
    models_dir = prev / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no models"):
        carry_forward_cognitive_models(prev, tmp_path / "experiment2")


def test_carry_forward_missing_model_file_raises(tmp_path):
    prev, manifest = _prev_experiment_with_export(tmp_path)
    (prev / "cognitive_models" / f"{manifest['models'][0]['name']}.py").unlink()
    with pytest.raises(FileNotFoundError, match=manifest["models"][0]["name"]):
        carry_forward_cognitive_models(prev, tmp_path / "experiment2")
