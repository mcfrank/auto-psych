"""Tests for the PyMC EIG/design annotator.

`annotate` scores candidate stimuli by expected information gain over the PyMC
model set, using prior-predictive p_left (no MCMC fit). It featurizes each raw
stimulus via the project's `featurize_stimulus` before handing it to the models.
Uses prior-predictive sampling (fast-ish, no NUTS) — marked slow to be safe.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop import eig as eig_mod

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"
FEATURIZE = (
    Path(__file__).resolve().parent.parent
    / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)

# A model with a participant-level random effect: it needs a `participant_id`
# pm.Data column that stimulus feature rows (n_a/h_a/...) never carry. It fits
# fine on responses.csv (which has participant_id) but cannot be prior-predicted
# on a bare stimulus — the operation EIG/design needs — so it must be screened
# out of the EIG model set rather than crashing the whole annotation. This is the
# shape that killed the impossible-holdout cell run4/more_imbalance.
_PARTICIPANT_MODEL = """import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
    u = pm.Normal("u", mu=0.0, sigma=sigma_u, shape=64)
    tau = pm.HalfNormal("tau", sigma=2.0)

    score = pt.cast(h_a - h_b, "float64")
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * score + u[participant_id])
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""


def _seed(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(FIXTURE_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(
        FIXTURE_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml"
    )
    return models_dir


def _seed_with_participant_model(tmp_path):
    """Seed set + one carried-forward model that requires a participant_id column."""
    models_dir = _seed(tmp_path)
    (models_dir / "participant_re.py").write_text(
        _PARTICIPANT_MODEL, encoding="utf-8"
    )
    manifest = yaml.safe_load(
        (models_dir / "models_manifest.yaml").read_text(encoding="utf-8")
    )
    manifest["models"].append(
        {"name": "participant_re", "rationale": "participant random effect."}
    )
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return models_dir


def test_annotate_drops_model_that_cannot_bind_to_stimulus(
    tmp_path, monkeypatch, capsys
):
    """A model needing a non-stimulus column (participant_id) is screened out of
    the EIG set — loudly — instead of crashing the whole annotation; the EIG runs
    over the remaining, stimulus-predictable models. EIG is stubbed so the screen
    (a real load + make_stim_data bind check) is what's exercised, not sampling."""
    models_dir = _seed_with_participant_model(tmp_path)

    seen: dict = {}

    def fake_eig(feature_row, model_names, models_dir, **kwargs):
        seen["names"] = list(model_names)
        return 0.5

    monkeypatch.setattr(
        "src.models.pymc_inference.expected_information_gain_prior_pymc", fake_eig
    )

    out = eig_mod.annotate(
        [{"sequence_a": "HHHT", "sequence_b": "HTHT"}],
        models_dir,
        featurize_path=FEATURIZE,
    )

    assert out[0]["eig"] == 0.5
    # The participant-requiring model was screened out before EIG; the others stay.
    assert "participant_re" not in seen["names"]
    assert "bayesian_fair_coin" in seen["names"]
    assert "representativeness" in seen["names"]
    assert "participant_re" in capsys.readouterr().out  # dropped loudly


def test_annotate_raises_when_no_model_can_bind(tmp_path):
    """If no model can be evaluated on a stimulus, fail loudly rather than emit
    meaningless EIGs."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "participant_re.py").write_text(
        _PARTICIPANT_MODEL, encoding="utf-8"
    )
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "participant_re", "rationale": "p."}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no models|cannot be evaluated|stimulus"):
        eig_mod.annotate(
            [{"sequence_a": "HHHT", "sequence_b": "HTHT"}],
            models_dir,
            featurize_path=FEATURIZE,
        )


@pytest.mark.slow
def test_annotate_adds_nonnegative_eig_and_sorts(tmp_path):
    models_dir = _seed(tmp_path)
    candidates = [
        {"sequence_a": "HHHHH", "sequence_b": "HHHHH"},  # identical → low EIG
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"},  # discriminating
    ]
    out = eig_mod.annotate(
        candidates, models_dir, featurize_path=FEATURIZE, n_samples=100
    )

    assert len(out) == 2
    for item in out:
        assert "eig" in item
        assert 0.0 <= item["eig"] <= 1.0  # 2 models → ≤ log2(2) = 1 bit
        assert "sequence_a" in item and "sequence_b" in item
    # Sorted descending by EIG.
    assert out[0]["eig"] >= out[1]["eig"]


@pytest.mark.slow
def test_annotate_featurizes_so_models_can_read_columns(tmp_path):
    """Without featurization the models' pm.Data columns are absent → this
    proves the annotator derives n_a/h_a/... from raw sequences."""
    models_dir = _seed(tmp_path)
    candidates = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    out = eig_mod.annotate(
        candidates, models_dir, featurize_path=FEATURIZE, n_samples=100
    )
    assert out[0]["eig"] >= 0.0


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "cognitive_models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        eig_mod.annotate(
            [{"sequence_a": "H", "sequence_b": "T"}], tmp_path / "cognitive_models"
        )
