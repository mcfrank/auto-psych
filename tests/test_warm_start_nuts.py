"""``fit.method: warm_start_nuts`` (``pymc_inference._warm_start_sample``):
NUTS started from the posterior mode, with the inverse Hessian as its mass
matrix. A fit setting like ``draws``, in the cache key only when it is not
the default ``nuts``, so every existing fingerprint is unchanged.
"""

from __future__ import annotations

import shutil

import pytest

from src.models import pymc_inference as pi
from tests.paths import PYMC_MODEL_FIXTURES_DIR


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(PYMC_MODEL_FIXTURES_DIR, d)
    return d


def test_the_default_method_leaves_every_existing_fingerprint_unchanged(models_dir):
    default = pi.resolve_fit_settings("representativeness", models_dir, {})
    assert default["method"] == "nuts"
    assert "method" not in pi._sampler_signature(default)
    warm = pi.resolve_fit_settings("representativeness", models_dir, {"method": "warm_start_nuts"})
    assert pi._sampler_signature(warm) == pi._sampler_signature(default) + ";method=warm_start_nuts"


def test_an_unknown_method_is_refused(models_dir):
    with pytest.raises(ValueError, match="method"):
        pi.resolve_fit_settings("representativeness", models_dir, {"method": "laplace"})


def test_a_model_file_cannot_choose_the_method(models_dir):
    source = models_dir / "representativeness.py"
    source.write_text(source.read_text() + '\nSAMPLER_SETTINGS = {"method": "warm_start_nuts"}\n')
    with pytest.raises((ValueError, TypeError), match="method"):
        pi.resolve_fit_settings("representativeness", models_dir, {})


@pytest.mark.slow
def test_a_warm_start_fit_is_a_converged_nuts_fit(models_dir, tmp_path):
    responses = models_dir / "responses.csv"
    fitted = pi.fit_model(
        "representativeness", models_dir, responses, cache_dir=tmp_path / "cache",
        method="warm_start_nuts", tune=200, draws=250,
    )
    assert fitted.idata.posterior.attrs["fit_method"] == "warm_start_nuts"
    assert pi.convergence_problems_of(fitted) == []  # divergences, R-hat and ESS, as for any fit
    assert "log_likelihood" in fitted.idata
    # Cached under its own fingerprint, apart from a default fit's.
    assert fitted.fingerprint != pi.fit_fingerprint(
        "representativeness", models_dir, responses, pi.resolve_fit_settings("representativeness", models_dir, {})
    )
    again = pi.fit_model(
        "representativeness", models_dir, responses, cache_dir=tmp_path / "cache",
        method="warm_start_nuts", tune=200, draws=250,
    )
    assert again.fingerprint == fitted.fingerprint
