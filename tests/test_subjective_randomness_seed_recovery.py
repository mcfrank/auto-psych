"""Fast tests for subjective-randomness seed and recovery helpers."""

from __future__ import annotations

import yaml

from src.models.model_loading import load_pymc_model, observed_response_data
from src.pipelines.outer_loop.orchestrator import (
    get_ground_truth_models,
    seed_experiment_models_from_project,
)
from tests.paths import REPO_ROOT

REGISTRY_DIR = REPO_ROOT / "src/subjective_randomness/pymc_model_families"


def test_seed_experiment_models_from_project_copies_active_seed_set(tmp_path):
    exp_dir = tmp_path / "experiment1"

    assert seed_experiment_models_from_project(exp_dir, "subjective_randomness")
    assert not seed_experiment_models_from_project(exp_dir, "subjective_randomness")

    models_dir = exp_dir / "cognitive_models"
    manifest = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    names = [m["name"] for m in manifest["models"]]

    # The live seed pool mirrors the registry manifest, which is the single
    # source of truth for the active seed set (the literature-faithful four).
    # The two diverged once — the hero-run winners were promoted into the pool
    # in 2026-07, the faithful set consolidated into the registry in 2026-08 —
    # and every recovery helper that resolves a seed name to its pure-Python
    # twin broke silently. Asserting equality here is the tripwire.
    registry = yaml.safe_load((REGISTRY_DIR / "models_manifest.yaml").read_text())
    assert names == [m["name"] for m in registry["models"]]
    for name in names:
        model = load_pymc_model(name, models_dir)
        assert observed_response_data(model) == "chose_left"


def test_hidden_ground_truth_models_return_valid_probabilities():
    registry = get_ground_truth_models("subjective_randomness")
    assert "length_sensitive_alternation" in registry
    assert "recency_weighted_alternation" in registry

    for name in ("length_sensitive_alternation", "recency_weighted_alternation"):
        probs = registry[name](("HHHT", "HTHT"), ["left", "right"])
        assert set(probs) == {"left", "right"}
        assert 0.0 <= probs["left"] <= 1.0
        assert abs((probs["left"] + probs["right"]) - 1.0) < 1e-12
