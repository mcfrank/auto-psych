"""Fast tests for subjective-randomness seed and recovery helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from src.models.pymc_inference import load_pymc_model, observed_response_data
from src.pipelines.outer_loop.orchestrator import (
    get_ground_truth_models,
    seed_experiment_models_from_project,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = (
    REPO_ROOT
    / "src/pipelines/outer_loop/projects/subjective_randomness/evaluate_recovery.py"
)


def _load_eval():
    spec = importlib.util.spec_from_file_location("_sr_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_experiment_models_from_project_copies_active_seed_set(tmp_path):
    exp_dir = tmp_path / "experiment1"

    assert seed_experiment_models_from_project(exp_dir, "subjective_randomness")
    assert not seed_experiment_models_from_project(exp_dir, "subjective_randomness")

    models_dir = exp_dir / "cognitive_models"
    manifest = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    names = [m["name"] for m in manifest["models"]]

    assert names == [
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
    ]
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


def test_recovery_eval_helpers_are_deterministic_and_metric_sensitive():
    ev = _load_eval()

    assert ev.parse_experiments("1-3") == [1, 2, 3]
    assert ev.parse_experiments("1,3") == [1, 3]

    stimuli_a = ev.make_heldout_stimuli(n_pairs=5, min_length=4, max_length=5, seed=10)
    stimuli_b = ev.make_heldout_stimuli(n_pairs=5, min_length=4, max_length=5, seed=10)
    assert stimuli_a == stimuli_b
    assert len(stimuli_a) == 5

    rows = ev.feature_rows(stimuli_a)
    assert {"sequence_a", "sequence_b", "chose_left", "imbalance_a"} <= set(rows[0])

    true = np.array([0.2, 0.8])
    good = ev.metrics(true, np.array([0.25, 0.75]))
    bad = ev.metrics(true, np.array([0.8, 0.2]))
    assert good["rmse"] < bad["rmse"]
    assert good["cross_entropy"] < bad["cross_entropy"]
