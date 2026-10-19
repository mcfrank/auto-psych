"""``fit.method: nuts_no_jitter``: production NUTS with every chain started at
the model's initial point (PyMC's ``init="adapt_diag"``) instead of a jittered
one. On the paper's pooled subjective-randomness data (23,104 trials) jittered
starts left 3 of 56 chains where the step size fell to zero in warm-up, so the
chain never moved (2026-10-18). In the cache key only when chosen, like every
method other than the default ``nuts``.
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


def test_no_jitter_fits_have_a_cache_key_of_their_own(models_dir):
    default = pi.resolve_fit_settings("representativeness", models_dir, {})
    plain = pi.resolve_fit_settings("representativeness", models_dir, {"method": "nuts_no_jitter"})
    assert pi._sampler_signature(plain) == pi._sampler_signature(default) + ";method=nuts_no_jitter"


def test_no_jitter_fits_start_every_chain_at_the_initial_point(models_dir, tmp_path, monkeypatch):
    pm = pi._import_pymc()
    seen = {}

    def fake_sample(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before sampling")

    monkeypatch.setattr(pm, "sample", fake_sample)
    with pytest.raises(RuntimeError, match="stop before sampling"):
        pi.fit_model("representativeness", models_dir, models_dir / "responses.csv",
                     method="nuts_no_jitter", tune=10, draws=10)
    assert seen["init"] == "adapt_diag"


def test_default_fits_keep_pymcs_jittered_start(models_dir, monkeypatch):
    pm = pi._import_pymc()
    seen = {}

    def fake_sample(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before sampling")

    monkeypatch.setattr(pm, "sample", fake_sample)
    with pytest.raises(RuntimeError, match="stop before sampling"):
        pi.fit_model("representativeness", models_dir, models_dir / "responses.csv", tune=10, draws=10)
    assert seen.get("init", "auto") == "auto"
