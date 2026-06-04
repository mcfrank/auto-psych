"""Fast unit tests for src/model_comparison/posterior.py.

The ELPD-LOO scoring (which needs MCMC) is monkeypatched out so these tests
exercise only the softmax/posterior arithmetic and the complexity prior — no
PyMC sampling required.
"""
from __future__ import annotations

import math

import pytest

from src.model_comparison import likelihood as ll
from src.model_comparison import posterior as mp


def _make_models_dir(tmp_path, names):
    """Create a models_dir with empty <name>.py files + a manifest listing them."""
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir()
    for name in names:
        (models_dir / f"{name}.py").write_text("# placeholder model\n", encoding="utf-8")
    manifest = "models:\n" + "".join(f"  - name: {n}\n" for n in names)
    (models_dir / "models_manifest.yaml").write_text(manifest, encoding="utf-8")
    return models_dir


def _make_responses(tmp_path, n_rows):
    csv_path = tmp_path / "responses.csv"
    rows = "\n".join("1,0" for _ in range(n_rows))
    csv_path.write_text("chose_left,x\n" + rows + "\n", encoding="utf-8")
    return csv_path


def test_posteriors_softmax_elpd_and_sum_to_one(tmp_path, monkeypatch):
    models_dir = _make_models_dir(tmp_path, ["good", "bad"])
    responses = _make_responses(tmp_path, 5)
    elpd = {"good": -10.0, "bad": -12.0}
    monkeypatch.setattr(ll, "log_likelihood", lambda m, *a, **k: elpd[m])

    result = mp.model_posterior(responses, models_dir)

    assert set(result["posteriors"]) == {"good", "bad"}
    assert pytest.approx(1.0) == sum(result["posteriors"].values())
    assert result["elpd_loo"] == {"good": -10.0, "bad": -12.0}
    assert result["n_trials"] == 5
    # 2-nat advantage → softmax weight e^2 : 1
    expected_good = math.exp(2.0) / (math.exp(2.0) + 1.0)
    assert result["posteriors"]["good"] == pytest.approx(expected_good, abs=1e-5)
    assert result["posteriors"]["good"] > result["posteriors"]["bad"]


def test_complexity_prior_penalises_longer_models(tmp_path, monkeypatch):
    models_dir = _make_models_dir(tmp_path, ["simple", "complex"])
    # Make "complex" genuinely longer so model_complexity differs.
    (models_dir / "complex.py").write_text("x = 1\ny = 2\nz = 3\n" * 50, encoding="utf-8")
    responses = _make_responses(tmp_path, 3)
    # Equal fit; only the complexity prior should break the tie.
    monkeypatch.setattr(ll, "log_likelihood", lambda m, *a, **k: -10.0)

    result = mp.model_posterior(responses, models_dir, complexity_prior_const=-0.01)

    assert result["complexity_prior_const"] == -0.01
    assert result["complexities"]["complex"] > result["complexities"]["simple"]
    assert result["posteriors"]["simple"] > result["posteriors"]["complex"]


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "cognitive_models").mkdir()
    with pytest.raises(FileNotFoundError):
        mp.model_posterior(tmp_path / "responses.csv", tmp_path / "cognitive_models")
