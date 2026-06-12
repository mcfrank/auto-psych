"""Tests for the end-to-end recovery pipeline (`src/subjective_randomness/pipeline.py`).

One call runs Bayesian (PyMC) parameter recovery for every model-family
config, optionally the stimulus-selection comparison and the closed-ended
model recovery, and writes all artifacts (reports, summary CSVs, figures) plus
an aggregated `key_results.txt` into a single output directory. MCMC itself is
monkeypatched (`fit_model` returns a canned posterior; the model-recovery
runner returns a canned confusion); everything around it runs for real.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.subjective_randomness.pipeline import key_results_text, run_pipeline

STIMULI = [
    {"sequence_a": "HHHHTTTT", "sequence_b": "HTHTHTHT"},
    {"sequence_a": "HHTHTTHT", "sequence_b": "HHHHHHHH"},
    {"sequence_a": "HTTHTHHT", "sequence_b": "HHHTTTHH"},
    {"sequence_a": "HTHTHTHH", "sequence_b": "TTTTHHHH"},
]


class _FakeParam:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)


class _FakeIdata:
    def __init__(self, params):
        self.posterior = {name: _FakeParam(values) for name, values in params.items()}


class _FakeFitted:
    fingerprint = "fake-fit"

    def __init__(self):
        self.idata = _FakeIdata(
            {
                "theta_alt": [[0.60, 0.70], [0.65, 0.67]],
                "alt_weight": [[0.50, 0.55], [0.58, 0.57]],
                "beta": [[3.8, 4.2], [4.0, 4.1]],
                "side_bias": [[-0.1, 0.0], [0.1, 0.0]],
            }
        )


@pytest.fixture
def fake_fit_model(monkeypatch):
    """Replace MCMC with a canned posterior; capture the sampling settings."""
    captured = {}

    def _fake(name, **kwargs):
        captured.update(kwargs)
        return _FakeFitted()

    monkeypatch.setattr("src.models.pymc_inference.fit_model", _fake)
    return captured

CANNED_CONFUSION = {
    "seed_models": ["A", "B"],
    "generator": "pymc",
    "generating": [
        {
            "generating_model": "A",
            "params": {},
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.8, "B": 0.2},
            "elpd_loo": {"A": -1.0, "B": -2.0},
        },
        {
            "generating_model": "B",
            "params": {},
            "best_model": "B",
            "recovered_correct": True,
            "posteriors": {"A": 0.3, "B": 0.7},
            "elpd_loo": {"A": -2.0, "B": -1.0},
        },
    ],
}


def _param_config(tmp_path: Path) -> Path:
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(json.dumps(STIMULI), encoding="utf-8")
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "simulation": {"n_repeats": 3, "n_participants": 6, "seed": 7},
        "mcmc": {"draws": 8, "tune": 4, "chains": 1},
    }
    path = tmp_path / "prototype_similarity.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")  # JSON is valid YAML
    return path


def _model_recovery_config(tmp_path: Path) -> Path:
    path = tmp_path / "model_recovery.yaml"
    path.write_text("seed_models_dir: .\n", encoding="utf-8")
    return path


def test_run_pipeline_writes_reports_figures_and_key_results(
    tmp_path, monkeypatch, fake_fit_model
):
    monkeypatch.setattr(
        "src.subjective_randomness.model_recovery.run_recovery_from_config",
        lambda *args, **kwargs: CANNED_CONFUSION,
    )
    out_dir = tmp_path / "out"

    result = run_pipeline(
        [_param_config(tmp_path)],
        out_dir,
        _model_recovery_config(tmp_path),
    )

    # Parameter recovery is Bayesian: each run holds a posterior summary, and
    # the config's MCMC settings reach the sampler.
    report = json.loads(
        (out_dir / "prototype_similarity_recovery.json").read_text(encoding="utf-8")
    )
    assert report["model"] == "prototype_similarity"
    assert len(report["runs"]) == 3
    assert "posterior" in report["runs"][0]
    assert "true_params" in report["runs"][0]  # sampled-truth mode
    assert fake_fit_model["draws"] == 8
    assert fake_fit_model["tune"] == 4
    assert fake_fit_model["chains"] == 1
    assert (out_dir / "prototype_similarity_recovery_tidy.csv").exists()
    with (out_dir / "prototype_similarity_summary.csv").open(encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert "pearson_r" in summary_rows[0]
    assert (out_dir / "prototype_similarity_correlation.png").stat().st_size > 0

    # Model-recovery artifacts.
    saved = json.loads((out_dir / "confusion.json").read_text(encoding="utf-8"))
    assert saved == CANNED_CONFUSION
    assert (out_dir / "confusion.csv").exists()
    assert (out_dir / "confusion.png").stat().st_size > 0

    # Aggregated key results.
    text = (out_dir / "key_results.txt").read_text(encoding="utf-8")
    assert "Parameter recovery — model: prototype_similarity" in text
    assert "pearson_r" in text
    assert "Closed-ended model recovery" in text
    assert result["key_results_path"] == out_dir / "key_results.txt"
    assert result["confusion"] == CANNED_CONFUSION
    assert [r["model"] for r in result["reports"]] == ["prototype_similarity"]


def test_run_pipeline_without_model_recovery_config_skips_that_stage(
    tmp_path, fake_fit_model
):
    out_dir = tmp_path / "out"

    result = run_pipeline([_param_config(tmp_path)], out_dir, None)

    assert not (out_dir / "confusion.json").exists()
    assert result["confusion"] is None
    text = (out_dir / "key_results.txt").read_text(encoding="utf-8")
    assert "model recovery: skipped" in text.lower()


def test_run_pipeline_mcmc_overrides_beat_the_config(tmp_path, fake_fit_model):
    run_pipeline(
        [_param_config(tmp_path)],
        tmp_path / "out",
        None,
        fit_overrides={"draws": 3, "tune": 2},
    )

    assert fake_fit_model["draws"] == 3  # CLI override wins
    assert fake_fit_model["tune"] == 2
    assert fake_fit_model["chains"] == 1  # untouched settings fall back to config


def _selection_comparison_config(tmp_path: Path) -> Path:
    config = {
        "model_names": ["prototype_similarity", "encoding_compressibility"],
        "pool": {"n_pairs": 20, "lengths": [6]},
        "n_stimuli": 4,
        "points_per_dim": 3,
        "seed": 0,
        "parameter_repeats": 2,
        "parameter_participants": 10,
        "model_repeats": 1,
        "model_participants": 4,
    }
    path = tmp_path / "selection_comparison.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")  # JSON is valid YAML
    return path


def test_run_pipeline_with_selection_comparison_writes_artifacts_and_text(
    tmp_path, fake_fit_model
):
    out_dir = tmp_path / "out"

    result = run_pipeline(
        [_param_config(tmp_path)],
        out_dir,
        None,
        selection_comparison_config_path=_selection_comparison_config(tmp_path),
    )

    # Per-family EIG-vs-random stimulus-set comparison artifacts.
    report = json.loads(
        (out_dir / "selection_comparison_prototype_similarity.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(report["arms"]) == {"eig", "random"}
    assert len(report["arms"]["eig"]["stimuli"]) == 4
    assert len(report["arms"]["eig"]["runs"]) == 2
    assert report["n_participants"] == 10  # parameter_participants
    assert (
        out_dir / "selection_comparison_prototype_similarity.png"
    ).stat().st_size > 0
    assert (out_dir / "selection_comparison_encoding_compressibility.json").exists()

    # Model-identity comparison artifacts, with their own data budget.
    model_report = json.loads(
        (out_dir / "selection_comparison_model_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(model_report["arms"]) == {"eig", "random"}
    assert model_report["n_participants"] == 4  # model_participants
    assert (out_dir / "selection_comparison_model_recovery.png").stat().st_size > 0

    # Both comparisons land in the aggregated key results.
    text = (out_dir / "key_results.txt").read_text(encoding="utf-8")
    assert "Stimulus-selection comparison — model: prototype_similarity" in text
    assert "Model-recovery stimulus-selection comparison" in text
    comparison = result["selection_comparison"]
    assert [r["model"] for r in comparison["parameter"]] == [
        "prototype_similarity",
        "encoding_compressibility",
    ]
    assert comparison["model"]["model_names"] == [
        "prototype_similarity",
        "encoding_compressibility",
    ]


def test_run_pipeline_without_selection_comparison_notes_skip(
    tmp_path, fake_fit_model
):
    out_dir = tmp_path / "out"

    result = run_pipeline([_param_config(tmp_path)], out_dir, None)

    assert result["selection_comparison"] is None
    text = (out_dir / "key_results.txt").read_text(encoding="utf-8")
    assert "selection comparison: skipped" in text.lower()


def test_run_pipeline_rejects_empty_config_list(tmp_path):
    with pytest.raises(ValueError, match="parameter-recovery config"):
        run_pipeline([], tmp_path / "out", None)


def test_key_results_text_orders_sections_and_notes_skip():
    report = {
        "model": "demo",
        "n_repeats": 2,
        "param_ranges": {"beta": [0.2, 12.0]},
        "runs": [
            {
                "repeat": 0,
                "true_params": {"beta": 1.0},
                "posterior": {"beta": {"mean": 1.2, "q025": 0.8, "q975": 1.6}},
            },
            {
                "repeat": 1,
                "true_params": {"beta": 8.0},
                "posterior": {"beta": {"mean": 7.5, "q025": 6.9, "q975": 8.1}},
            },
        ],
    }

    text = key_results_text([report], None)

    assert text.index("key results") < text.index("Parameter recovery")
    assert "model recovery: skipped" in text.lower()
