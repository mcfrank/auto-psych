"""Pruning uses the stimulus-clustered standard error of the ELPD difference.

Every experiment shows the same pairs to all its participants, so the trials
on one pair favour the same model together; az.compare's standard error
treats every trial as independent and is ~2x too small (docs: the LOO design
effect memo). The clustered SE sums the pointwise differences within each
stimulus — an unordered pair, since left/right is counterbalanced — and takes
the standard error over stimuli (user decision 2026-09-26).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.clustered_se import cluster_dse, stimulus_clusters
from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.model_zoo import _prune_losers


def test_both_presentations_of_a_pair_are_one_stimulus():
    rows = [
        {"sequence_a": "HT", "sequence_b": "HH"},
        {"sequence_a": "HH", "sequence_b": "HT"},
        {"sequence_a": "TT", "sequence_b": "HH"},
    ]
    groups = stimulus_clusters(rows)
    assert groups[0] == groups[1] != groups[2]


def test_the_clustered_se_is_the_se_over_stimulus_sums():
    diff_best, diff_other = np.array([1.0, 1.0, 0.0, 0.0]), np.zeros(4)
    groups = np.array([0, 0, 1, 1])
    # Stimulus sums 2 and 0: sqrt(G * var) = sqrt(2 * 1) .
    assert cluster_dse(diff_best, diff_other, groups) == pytest.approx(np.sqrt(2.0))


def test_correlated_trials_make_the_clustered_se_larger():
    rng = np.random.default_rng(0)
    per_stimulus = rng.normal(0, 1, size=64)
    diff = np.repeat(per_stimulus, 40) + rng.normal(0, 0.1, size=64 * 40)
    groups = np.repeat(np.arange(64), 40)
    trial_se = np.sqrt(diff.size * diff.var())
    assert cluster_dse(diff, np.zeros_like(diff), groups) > 3 * trial_se


def _zoo(tmp_path):
    import yaml

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    names = ["best", "loser"]
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": n, "rationale": f"H {n}"} for n in names]}),
        encoding="utf-8",
    )
    for n in names:
        (models_dir / f"{n}.py").write_text("# model\n", encoding="utf-8")
    return models_dir


def test_a_model_behind_by_2_trial_ses_but_not_2_clustered_ses_is_kept(tmp_path, monkeypatch):
    comparison = {
        "best": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "dse_clustered": 0.0},
        "loser": {"rank": 1, "elpd_diff": 10.0, "dse": 3.0, "dse_clustered": 6.0},
    }
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: comparison)
    monkeypatch.setattr(model_zoo, "evict_fit_cache", lambda name: None)
    assert _prune_losers(
        _zoo(tmp_path), tmp_path / "r.csv", protected=set(), cache_dir=None, fit_kwargs={}
    ) == []


def test_pruning_without_a_clustered_se_fails_loudly(tmp_path, monkeypatch):
    comparison = {
        "best": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0},
        "loser": {"rank": 1, "elpd_diff": 10.0, "dse": 3.0},
    }
    monkeypatch.setattr(model_zoo, "compare_table", lambda *a, **k: comparison)
    with pytest.raises(KeyError, match="dse_clustered"):
        _prune_losers(
            _zoo(tmp_path), tmp_path / "r.csv", protected=set(), cache_dir=None, fit_kwargs={}
        )
