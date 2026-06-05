"""Fast tests for the az.compare-based distinguishability diagnostic.

`compare_table` ranks the fitted models by ELPD-LOO and reports, for each, the
ELPD difference from the best model and the standard error of that difference
(`dse`) — so a reader can tell whether the top models are genuinely
distinguishable or within noise. MCMC is monkeypatched out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.model_comparison import likelihood as ll
from src.model_comparison import posterior as mp


def _seed_models(tmp_path, names):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir()
    for name in names:
        (models_dir / f"{name}.py").write_text("# placeholder\n", encoding="utf-8")
    manifest = "models:\n" + "".join(f"  - name: {n}\n" for n in names)
    (models_dir / "models_manifest.yaml").write_text(manifest, encoding="utf-8")
    return models_dir


def _fake_idata(mean_ll, n_obs=25, n_chains=4, n_draws=250, seed=0):
    """InferenceData with a log_likelihood group shaped for az.loo/az.compare.

    Per-observation log-likelihood centred at ``mean_ll`` with small posterior
    spread — enough draws for PSIS-LOO to run.
    """
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(seed)
    # (chain, draw, obs); slight per-draw jitter so importance sampling is stable.
    arr = mean_ll + rng.normal(0.0, 0.02, size=(n_chains, n_draws, n_obs))
    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
    ll_ds = xr.Dataset(
        {"response": (("chain", "draw", "response_dim_0"), arr)},
        coords={**coords, "response_dim_0": np.arange(n_obs)},
    )
    # az.loo in this arviz version also requires a posterior group.
    post_ds = xr.Dataset(
        {"theta": (("chain", "draw"), rng.normal(0.0, 1.0, size=(n_chains, n_draws)))},
        coords=coords,
    )
    return az.InferenceData(posterior=post_ds, log_likelihood=ll_ds)


def test_compare_table_reports_elpd_diff_and_dse(tmp_path, monkeypatch):
    models_dir = _seed_models(tmp_path, ["good", "bad"])
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,x\n" + "1,0\n" * 25, encoding="utf-8")

    # "good" assigns higher per-obs log-likelihood than "bad".
    idata = {
        "good": _fake_idata(-0.3, seed=1),
        "bad": _fake_idata(-0.9, seed=2),
    }

    class _FakeFit:
        def __init__(self, name):
            self.idata = idata[name]

    def _fake_fit_models_cached(
        model_names, models_dir, responses_path, cache_dir=None, **kw
    ):
        return {m: _FakeFit(m) for m in model_names}

    monkeypatch.setattr(
        "src.models.pymc_inference.fit_models_cached", _fake_fit_models_cached
    )

    table = mp.compare_table(responses, models_dir)

    assert set(table) == {"good", "bad"}
    # Best model has zero elpd_diff and zero dse against itself.
    assert table["good"]["elpd_diff"] == pytest.approx(0.0, abs=1e-9)
    assert table["good"]["dse"] == pytest.approx(0.0, abs=1e-9)
    # The worse model is behind (positive elpd_diff) with a real SE.
    assert table["bad"]["elpd_diff"] > 0
    assert table["bad"]["dse"] > 0
    assert "rank" in table["good"] and table["good"]["rank"] == 0
