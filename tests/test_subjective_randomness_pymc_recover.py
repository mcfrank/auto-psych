"""Tests for the subjective-randomness PyMC recovery bridge."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.subjective_randomness import pymc_recover


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


def test_posterior_summary_extracts_requested_params():
    idata = _FakeIdata({"alpha": [[0.0, 1.0], [2.0, 3.0]], "ignored": [[10.0]]})
    summary = pymc_recover.posterior_summary(idata, ["alpha", "missing"])
    assert set(summary) == {"alpha"}
    assert summary["alpha"]["mean"] == 1.5
    assert summary["alpha"]["q025"] == 0.075
    assert summary["alpha"]["q975"] == 2.925


def test_featurize_response_rows_adds_pymc_columns():
    rows = [
        {
            "participant_id": 0,
            "trial_index": 0,
            "sequence_a": "HTHT",
            "sequence_b": "HHHT",
            "chose_left": 1,
        }
    ]
    [row] = pymc_recover.featurize_response_rows(rows)
    assert row["p_alts_a"] == 1.0
    assert row["imbalance_a"] == 0.0
    assert row["periodicity_a"] == 1.0
    assert row["max_run_norm_b"] == 2.0 / 3.0


def test_run_pymc_recovery_writes_featurized_rows_and_report(tmp_path, monkeypatch):
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text('[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config = {
        "model_module": "cc_pipeline.projects.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "true_params": {
            "theta_alt": 0.65,
            "alt_weight": 0.55,
            "beta": 4.0,
            "side_bias": 0.0,
        },
        "simulation": {"n_participants": 2, "n_repeats": 1, "seed": 7},
    }
    work_dir = tmp_path / "work"
    captured = {}

    def fake_fit_model(name, models_dir, responses_path, **kwargs):
        captured["name"] = name
        captured["models_dir"] = Path(models_dir)
        captured["responses_path"] = Path(responses_path)
        captured["kwargs"] = kwargs
        return _FakeFitted()

    monkeypatch.setattr("src.models.pymc_inference.fit_model", fake_fit_model)

    result = pymc_recover.run_pymc_recovery(
        config,
        config_path,
        work_dir=work_dir,
        draws=10,
        tune=5,
        chains=1,
        cores=1,
    )

    assert result["model"] == "prototype_similarity"
    assert result["n_repeats"] == 1
    assert result["summary"]["theta_alt"]["true"] == 0.65
    assert result["summary"]["theta_alt"]["mean_posterior_mean"] == 0.655
    assert result["runs"][0]["fit_fingerprint"] == "fake-fit"

    assert captured["name"] == "prototype_similarity"
    assert captured["kwargs"]["draws"] == 10
    assert captured["kwargs"]["tune"] == 5
    assert captured["kwargs"]["chains"] == 1

    with captured["responses_path"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "p_alts_a" in rows[0]
    assert "imbalance_b" in rows[0]
    assert rows[0]["model"] == "prototype_similarity"
