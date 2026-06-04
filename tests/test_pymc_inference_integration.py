"""End-to-end integration test for PyMC-based model comparison.

Verifies that, given two PyMC cognitive models and a responses CSV simulated
from one of them, `model_posterior` correctly identifies the data-generating
model with >0.7 posterior mass.

Slow (two NUTS fits on ~30 trials): ~10-30 s. Run with `pytest -m slow`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"


@pytest.mark.slow
def test_model_posterior_selects_data_generating_model():
    from src.models.pymc_inference import clear_fit_cache
    from src.model_comparison.posterior import model_posterior

    responses_path = FIXTURE_DIR / "responses.csv"
    assert responses_path.exists(), (
        f"Fixture responses.csv missing — regenerate with "
        f"`python3 {FIXTURE_DIR / 'simulate.py'}`"
    )

    clear_fit_cache()
    result = model_posterior(
        responses_path=responses_path,
        models_dir=FIXTURE_DIR,
    )

    posteriors = result["posteriors"]
    assert set(posteriors.keys()) == {"bayesian_fair_coin", "representativeness"}
    assert abs(sum(posteriors.values()) - 1.0) < 1e-4, posteriors
    assert posteriors["bayesian_fair_coin"] > 0.7, (
        f"Expected bayesian_fair_coin (data-generating model) to win with >0.7 mass; "
        f"got posteriors={posteriors}, elpd_loo={result['elpd_loo']}"
    )
