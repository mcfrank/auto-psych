"""Cached, time-limited fits for the RSA loop."""

import os
import shutil

import pytest

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import (
    FitTimeLimitExceeded,
    ModelFailure,
    cache_paths,
    fingerprint,
    fit_cached,
)
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
QUICK = FitSettings(num_warmup=50, num_samples=50, num_chains=2)


@pytest.fixture
def responses(tmp_path):
    """A small responses file: the E8 levels experiment's included trials."""
    import pandas as pd

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    path = tmp_path / "responses.csv"
    df[df.experiment == "E8_levels"].to_csv(path, index=False)
    return path


def test_a_fit_is_cached_by_content(tmp_path, responses):
    cache = tmp_path / "cache"
    a = fit_cached(SEEDS / "rsa_l1.py", "rsa_l1", responses, QUICK, cache)
    fp = fingerprint(SEEDS / "rsa_l1.py", responses, QUICK)
    nc, meta = cache_paths(cache, "rsa_l1", fp)
    assert nc.exists() and meta.exists()
    b = fit_cached(SEEDS / "rsa_l1.py", "rsa_l1", responses, QUICK, cache)
    assert (a.idata.posterior["alpha"].values == b.idata.posterior["alpha"].values).all()
    assert b.param_names == ["alpha", "lapse"]
    # Different settings are a different fit.
    assert fingerprint(SEEDS / "rsa_l1.py", responses, FitSettings(num_warmup=51)) != fp


def test_a_model_error_in_a_time_limited_fit_is_a_model_failure(tmp_path, responses):
    bad = tmp_path / "broken.py"
    bad.write_text(
        (SEEDS / "rsa_l1.py").read_text().replace(
            'heard = L1(params["alpha"], ctx.lex)', 'heard = undefined_name'
        )
    )
    with pytest.raises(ModelFailure, match="NameError"):
        fit_cached(bad, "broken", responses, QUICK, tmp_path / "c", time_limit_sec=300)


def test_a_memo_compile_error_is_a_model_failure(tmp_path, responses):
    bad = tmp_path / "bare_eps.py"
    bad.write_text((SEEDS / "rsa_l1.py").read_text().replace("{EPS}", "EPS"))
    with pytest.raises(ModelFailure, match="MemoError"):
        fit_cached(bad, "bare_eps", responses, QUICK, tmp_path / "c")


def test_a_fit_over_the_time_limit_is_refused(tmp_path, responses):
    with pytest.raises(FitTimeLimitExceeded, match="did not finish"):
        fit_cached(SEEDS / "rsa_l2.py", "rsa_l2", responses, FitSettings(), tmp_path / "c",
                   time_limit_sec=1)


def test_a_time_limited_fit_imports_arviz_with_a_cache_of_its_own(tmp_path, responses, monkeypatch):
    # arviz writes a once-a-day marker into <XDG cache>/arviz through a fixed-name
    # temporary file; fit children that imported it together after midnight
    # collided there (three of Sherlock run 2's cells, 2026-10-08). A child must
    # not touch the parent's cache: here a marker arviz cannot read breaks any that does.
    shared = tmp_path / "shared_cache"
    (shared / "arviz" / "daily_warning").mkdir(parents=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(shared))
    fitted = fit_cached(SEEDS / "rsa_l1.py", "rsa_l1", responses, QUICK, tmp_path / "c", time_limit_sec=600)
    assert fitted.param_names
    assert os.environ["XDG_CACHE_HOME"] == str(shared)  # the parent's own is left as it was
