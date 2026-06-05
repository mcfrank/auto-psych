"""Fast unit tests for the subjective_randomness featurizer (no PyMC)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_sr_preprocess", PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_featurize_stimulus_counts_heads_alternations_runs():
    pp = _load()
    feats = pp.featurize_stimulus("HHHT", "HTHT")

    assert feats["n_a"] == 4 and feats["n_b"] == 4
    assert feats["h_a"] == 3 and feats["h_b"] == 2
    assert feats["alts_a"] == 1  # HHHT: one H→T transition
    assert feats["alts_b"] == 3  # HTHT: H-T-H-T, three transitions
    assert feats["max_run_a"] == 3  # "HHH"
    assert feats["max_run_b"] == 1
    assert feats["p_a"] == pytest.approx(0.75)
    assert feats["p_alts_b"] == pytest.approx(1.0)
    assert feats["imbalance_a"] == pytest.approx(0.5)
    assert feats["imbalance_b"] == pytest.approx(0.0)
    assert feats["max_run_norm_a"] == pytest.approx(2.0 / 3.0)
    assert feats["max_run_norm_b"] == pytest.approx(0.0)
    assert feats["periodicity_a"] == pytest.approx(0.5)
    assert feats["periodicity_b"] == pytest.approx(1.0)


def test_featurize_stimulus_keys_match_pm_data_names():
    pp = _load()
    feats = pp.featurize_stimulus("HT", "TH")
    expected = {
        "n_a",
        "h_a",
        "alts_a",
        "max_run_a",
        "p_a",
        "p_alts_a",
        "max_run_norm_a",
        "imbalance_a",
        "periodicity_a",
        "n_b",
        "h_b",
        "alts_b",
        "max_run_b",
        "p_b",
        "p_alts_b",
        "max_run_norm_b",
        "imbalance_b",
        "periodicity_b",
    }
    assert set(feats) == expected


def test_featurize_stimulus_handles_length_one():
    pp = _load()
    feats = pp.featurize_stimulus("H", "T")
    assert feats["alts_a"] == 0
    assert feats["p_alts_a"] == 0.0  # n-1 == 0 guard
    assert feats["p_a"] == 1.0
    assert feats["p_b"] == 0.0
    assert feats["max_run_norm_a"] == 0.0
    assert feats["periodicity_a"] == 0.0
