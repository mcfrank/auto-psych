"""Tests for the sampling-free fittability guard.

An agent can write a PyMC model that loads as a valid graph but whose logp
evaluates to NaN/-inf on the real data (e.g. the numerically unsafe
``pt.sqrt(x**2)``, which NaNs in PyTensor for some inputs). Such a model passes
graph-loading but crashes ``pm.sample`` at its start-value check, aborting the
whole run. ``model_logp_is_finite`` catches it cheaply, before sampling.
"""

from __future__ import annotations

from pathlib import Path

from src.models.pymc_inference import model_logp_is_finite

_GOOD = """
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    x_a = pm.Data("x_a", np.zeros(1, dtype="float64"))
    x_b = pm.Data("x_b", np.zeros(1, dtype="float64"))
    tau = pm.HalfNormal("tau", sigma=2.0)
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (pt.abs(x_b - 0.5) - pt.abs(x_a - 0.5)))
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""

# pt.sqrt((x - 0.5) ** 2) is mathematically |x - 0.5| but NaNs in PyTensor for
# inputs like 1/7 — exactly the alternation proportions of length-7 sequences.
_NAN = _GOOD.replace("pt.abs(x_b - 0.5)", "pt.sqrt((x_b - 0.5) ** 2)").replace(
    "pt.abs(x_a - 0.5)", "pt.sqrt((x_a - 0.5) ** 2)"
)


def _responses(tmp_path: Path) -> Path:
    p = tmp_path / "responses.csv"
    rows = "\n".join("0.142857,0.5,1" for _ in range(5))
    p.write_text("x_a,x_b,chose_left\n" + rows + "\n", encoding="utf-8")
    return p


def test_finite_logp_model_is_fittable(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "good.py").write_text(_GOOD, encoding="utf-8")
    ok, reason = model_logp_is_finite("good", models_dir, _responses(tmp_path))
    assert ok, reason


def test_nan_model_is_flagged_unfittable(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "bad.py").write_text(_NAN, encoding="utf-8")
    ok, reason = model_logp_is_finite("bad", models_dir, _responses(tmp_path))
    assert not ok
    assert "logp" in reason.lower()
