"""Tests for the subjective-randomness PyMC recovery bridge."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

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


def test_run_pymc_recovery_samples_truths_when_config_has_no_true_params(
    tmp_path, monkeypatch
):
    from src.subjective_randomness.model_families import prototype_similarity

    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        '[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8"
    )
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "simulation": {"n_participants": 2, "n_repeats": 3, "seed": 7},
    }
    monkeypatch.setattr(
        "src.models.pymc_inference.fit_model",
        lambda *args, **kwargs: _FakeFitted(),
    )

    result = pymc_recover.run_pymc_recovery(
        config, tmp_path / "config.yaml", draws=10, tune=5, chains=1
    )

    bounds = prototype_similarity.PARAM_BOUNDS
    # Every repeat carries its own ground truth inside the family bounds, and
    # the truths vary across repeats (sampled, not one fixed point).
    truths = [run["true_params"] for run in result["runs"]]
    assert len(truths) == 3
    for truth in truths:
        assert set(truth) == set(bounds)
        for name, value in truth.items():
            lo, hi = bounds[name]
            assert lo <= value <= hi
    assert any(truth != truths[0] for truth in truths[1:])
    # The sampled ranges are recorded, and the summary pairs each posterior
    # mean with its own run's truth.
    assert result["param_ranges"] == {
        name: [lo, hi] for name, (lo, hi) in bounds.items()
    }
    for name in bounds:
        assert "pearson_r" in result["summary"][name]
    assert "true_params" not in result


def test_run_pymc_recovery_uses_config_mcmc_block_when_no_kwargs_given(
    tmp_path, monkeypatch
):
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        '[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8"
    )
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "simulation": {"n_participants": 2, "n_repeats": 1, "seed": 7},
        "mcmc": {"draws": 12, "tune": 6, "chains": 3, "cores": 2},
    }
    captured = {}

    def fake_fit_model(name, models_dir, responses_path, **kwargs):
        captured.update(kwargs)
        return _FakeFitted()

    monkeypatch.setattr("src.models.pymc_inference.fit_model", fake_fit_model)

    result = pymc_recover.run_pymc_recovery(config, tmp_path / "config.yaml")

    assert captured["draws"] == 12
    assert captured["tune"] == 6
    assert captured["chains"] == 3
    assert captured["cores"] == 2
    # The report records what was actually sampled, not the defaults.
    assert (result["draws"], result["tune"], result["chains"]) == (12, 6, 3)


def test_run_pymc_recovery_explicit_kwargs_beat_config_mcmc_block(
    tmp_path, monkeypatch
):
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        '[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8"
    )
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "simulation": {"n_participants": 2, "n_repeats": 1, "seed": 7},
        "mcmc": {"draws": 12, "tune": 6, "chains": 3},
    }
    captured = {}

    def fake_fit_model(name, models_dir, responses_path, **kwargs):
        captured.update(kwargs)
        return _FakeFitted()

    monkeypatch.setattr("src.models.pymc_inference.fit_model", fake_fit_model)

    pymc_recover.run_pymc_recovery(config, tmp_path / "config.yaml", draws=10)

    assert captured["draws"] == 10  # the explicit kwarg wins
    assert captured["tune"] == 6  # untouched settings still come from the config


def test_run_pymc_recovery_rejects_unknown_mcmc_setting(tmp_path):
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": "unused.json",
        "mcmc": {"steps": 100},
    }
    with pytest.raises(ValueError, match="steps"):
        pymc_recover.run_pymc_recovery(config, tmp_path / "config.yaml")


def test_run_pymc_recovery_rejects_config_with_both_truth_modes(tmp_path):
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": "unused.json",
        "true_params": {"beta": 4.0},
        "param_ranges": {"beta": [1.0, 6.0]},
    }
    with pytest.raises(ValueError, match="pick one"):
        pymc_recover.run_pymc_recovery(config, tmp_path / "config.yaml")


def test_run_pymc_recovery_rejects_stale_fit_block(tmp_path):
    # A leftover `fit:` block holds settings for the deleted max-likelihood
    # optimizer; silently ignoring it would let a user believe those settings
    # still apply. Parameter fitting is Bayesian-only — fail loudly.
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": "unused.json",
        "fit": {"n_starts": 28, "max_iters": 180},
    }
    with pytest.raises(ValueError, match="mcmc"):
        pymc_recover.run_pymc_recovery(config, tmp_path / "config.yaml")


@pytest.mark.slow
def test_run_pymc_recovery_real_mcmc_fits_sampled_truths(tmp_path):
    # End-to-end with a real (tiny) NUTS run: sample truths, simulate, fit the
    # PyMC adapter, and summarize — no mocks anywhere.
    from src.subjective_randomness.model_families import prototype_similarity

    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        '[{"sequence_a": "HHHHTTTT", "sequence_b": "HTHTHTHT"},'
        ' {"sequence_a": "HHTHTTHT", "sequence_b": "HHHHHHHH"},'
        ' {"sequence_a": "HTTHTHHT", "sequence_b": "HHHTTTHH"}]',
        encoding="utf-8",
    )
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
        "stimuli_path": str(stimuli_path),
        "simulation": {"n_participants": 8, "n_repeats": 2, "seed": 11},
    }

    result = pymc_recover.run_pymc_recovery(
        config, tmp_path / "config.yaml", draws=80, tune=80, chains=1
    )

    bounds = prototype_similarity.PARAM_BOUNDS
    assert len(result["runs"]) == 2
    for run in result["runs"]:
        assert set(run["true_params"]) == set(bounds)
        assert set(run["posterior"]) == set(bounds)
        for name, entry in run["posterior"].items():
            # Posterior summaries are real numbers with a credible interval.
            assert entry["q025"] <= entry["mean"] <= entry["q975"]
    for name in bounds:
        assert "pearson_r" in result["summary"][name]


def test_run_pymc_recovery_writes_featurized_rows_and_report(tmp_path, monkeypatch):
    stimuli_path = tmp_path / "stimuli.json"
    stimuli_path.write_text(
        '[{"sequence_a": "HTHT", "sequence_b": "HHHT"}]', encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config = {
        "model_module": "src.subjective_randomness.model_families.prototype_similarity",
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
