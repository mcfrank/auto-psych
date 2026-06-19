"""Tests for theorist-extensible features in the PyMC inference bridge.

A candidate model may declare a module-level ``compute_features(sequence_a,
sequence_b) -> dict[str, float]`` to add numeric feature columns the base
featurizer never produced — e.g. order/position-sensitive statistics the fixed
feature set cannot express. The bridge computes them from the raw H/T sequences
(carried in responses.csv) and binds them to ``pm.Data`` containers by name.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from src.models import pymc_inference as pi

# A model whose single hypothesis needs a feature the base 11 cannot express:
# whether each sequence *ends* in H (a recency/position statistic, invisible to
# the order-destroying aggregate features).
ENDS_IN_H_MODEL = '''
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def ends_h(s):
        return 1.0 if s.strip().upper().endswith("H") else 0.0

    return {"ends_h_a": ends_h(sequence_a), "ends_h_b": ends_h(sequence_b)}


with pm.Model() as model:
    ends_h_a = pm.Data("ends_h_a", np.zeros(1, dtype="float64"))
    ends_h_b = pm.Data("ends_h_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (ends_h_a - ends_h_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
'''


def _write_model(models_dir: Path, source: str, name: str = "ends_in_h") -> str:
    (models_dir / f"{name}.py").write_text(source, encoding="utf-8")
    return name


def _write_raw_responses(csv_path: Path) -> None:
    """A responses CSV carrying the raw H/T sequences (as the real pipeline does)."""
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence_a", "sequence_b", "chose_left"])
        w.writeheader()
        w.writerow({"sequence_a": "HTH", "sequence_b": "HHT", "chose_left": "1"})
        w.writerow({"sequence_a": "TTT", "sequence_b": "HTH", "chose_left": "0"})


# ── Integration: the observable behavior, outside-in ───────────────────────────


def test_model_declared_feature_is_computed_from_raw_sequences(tmp_path):
    name = _write_model(tmp_path, ENDS_IN_H_MODEL)
    csv_path = tmp_path / "responses.csv"
    _write_raw_responses(csv_path)

    model = pi.load_pymc_model(name, tmp_path)
    observed = pi.extract_observed(csv_path, model)

    # The new columns exist and are computed from the raw sequences.
    assert observed["ends_h_a"].tolist() == [1.0, 0.0]  # HTH ends H, TTT ends T
    assert observed["ends_h_b"].tolist() == [0.0, 1.0]  # HHT ends T, HTH ends H
    assert observed["chose_left"].tolist() == [1, 0]


def test_model_with_declared_feature_has_finite_logp(tmp_path):
    name = _write_model(tmp_path, ENDS_IN_H_MODEL)
    csv_path = tmp_path / "responses.csv"
    _write_raw_responses(csv_path)

    fittable, reason = pi.model_logp_is_finite(name, tmp_path, csv_path)
    assert fittable, reason


# ── Unit: featurizer attachment + augmentation contract ────────────────────────


def test_load_attaches_declared_featurizer(tmp_path):
    name = _write_model(tmp_path, ENDS_IN_H_MODEL)
    model = pi.load_pymc_model(name, tmp_path)
    featurizer = pi._model_extra_featurizer(model)
    assert callable(featurizer)
    assert featurizer("HTH", "TTT") == {"ends_h_a": 1.0, "ends_h_b": 0.0}


def test_load_leaves_featurizer_none_when_absent():
    # The shipped seed/fixture model declares no compute_features.
    model = pi.load_pymc_model(
        "bayesian_fair_coin", Path(__file__).parent / "fixtures" / "pymc_models"
    )
    assert pi._model_extra_featurizer(model) is None


def test_load_rejects_non_callable_compute_features(tmp_path):
    source = ENDS_IN_H_MODEL.replace(
        "def compute_features(sequence_a, sequence_b):\n"
        '    def ends_h(s):\n'
        '        return 1.0 if s.strip().upper().endswith("H") else 0.0\n\n'
        '    return {"ends_h_a": ends_h(sequence_a), "ends_h_b": ends_h(sequence_b)}',
        "compute_features = 7",
    )
    name = _write_model(tmp_path, source, name="bad_featurizer")
    with pytest.raises(TypeError, match="compute_features"):
        pi.load_pymc_model(name, tmp_path)


def test_augment_is_noop_without_featurizer():
    import pymc as pm
    import numpy as np

    with pm.Model() as model:
        pm.Data("n_a", np.zeros(1, dtype="int64"))
    rows = [{"n_a": "3", "sequence_a": "HTH"}]
    assert pi._augment_rows_with_features(model, rows) == rows


def test_augment_requires_raw_sequences(tmp_path):
    name = _write_model(tmp_path, ENDS_IN_H_MODEL)
    model = pi.load_pymc_model(name, tmp_path)
    with pytest.raises(ValueError, match="sequence_a"):
        pi._augment_rows_with_features(model, [{"chose_left": "1"}])


def _model_from_featurizer(tmp_path, body: str, name: str):
    """Build the ends-in-h model but with a custom compute_features body."""
    source = (
        "import numpy as np\nimport pymc as pm\n\n"
        + body
        + "\n\nwith pm.Model() as model:\n"
        '    ends_h_a = pm.Data("ends_h_a", np.zeros(1, dtype="float64"))\n'
        '    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))\n'
        '    beta = pm.Uniform("beta", lower=0.2, upper=12.0)\n'
        '    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * ends_h_a))\n'
        '    pm.Bernoulli("response", p=p_left, observed=chose_left)\n'
    )
    return pi.load_pymc_model(_write_model(tmp_path, source, name=name), tmp_path)


def test_augment_rejects_non_numeric_feature(tmp_path):
    model = _model_from_featurizer(
        tmp_path,
        'def compute_features(a, b):\n    return {"ends_h_a": "yes"}',
        "nonnumeric",
    )
    with pytest.raises(ValueError, match="must be a number"):
        pi._augment_rows_with_features(
            model, [{"sequence_a": "H", "sequence_b": "T", "chose_left": "1"}]
        )


def test_augment_rejects_collision_with_existing_column(tmp_path):
    model = _model_from_featurizer(
        tmp_path,
        'def compute_features(a, b):\n    return {"chose_left": 1.0}',
        "collision",
    )
    with pytest.raises(ValueError, match="collides"):
        pi._augment_rows_with_features(
            model, [{"sequence_a": "H", "sequence_b": "T", "chose_left": "1"}]
        )


def test_augment_rejects_inconsistent_keys(tmp_path):
    model = _model_from_featurizer(
        tmp_path,
        'def compute_features(a, b):\n'
        '    return {"ends_h_a": 1.0} if a == "H" else {"other": 1.0}',
        "inconsistent",
    )
    rows = [
        {"sequence_a": "H", "sequence_b": "T", "chose_left": "1"},
        {"sequence_a": "TT", "sequence_b": "H", "chose_left": "0"},
    ]
    with pytest.raises(ValueError, match="inconsistent feature names"):
        pi._augment_rows_with_features(model, rows)


# ── End-to-end with real MCMC (the full fit + posterior-predictive path) ───────


@pytest.mark.slow
def test_custom_feature_model_fits_and_predicts_end_to_end(tmp_path):
    """A derived-feature model fits via MCMC and supports posterior-predictive draws.

    Exercises both augmentation seams under real sampling: extract_observed
    (during the fit) and make_stim_data (during synthetic-response sampling).
    """
    name = _write_model(tmp_path, ENDS_IN_H_MODEL)
    csv_path = tmp_path / "responses.csv"
    # People here favour the sequence whose last toss is heads — a recency cue the
    # base features cannot see. Build data consistent with that so MCMC has signal.
    sequences = ["HTH", "TTT", "HHT", "THH", "THT", "HTT"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence_a", "sequence_b", "chose_left"])
        w.writeheader()
        for i in range(36):
            a = sequences[i % len(sequences)]
            b = sequences[(i * 5 + 1) % len(sequences)]
            chose_left = int(a.endswith("H") and not b.endswith("H"))
            w.writerow({"sequence_a": a, "sequence_b": b, "chose_left": chose_left})

    fitted = pi.fit_model(
        name, tmp_path, csv_path, draws=200, tune=200, chains=2
    )

    stim_data = pi.make_stim_data(
        fitted.model, [{"sequence_a": "HTH", "sequence_b": "TTT", "chose_left": 0}]
    )
    assert stim_data["ends_h_a"].tolist() == [1.0]  # derived, not in the row

    p_left = fitted.predict_p_left(stim_data)
    assert p_left.shape == (1,)
    assert 0.0 < float(p_left[0]) < 1.0

    synthetic = fitted.sample_synthetic_responses(stim_data, n_datasets=20)
    assert synthetic.shape == (20, 1)
    assert set(np.unique(synthetic)).issubset({0, 1})
    assert math.isfinite(fitted.elpd_loo())
