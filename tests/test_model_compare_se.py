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


def _fake_idata(
    mean_ll,
    n_obs=25,
    n_chains=4,
    n_draws=250,
    seed=0,
    n_exact=0,
    n_heavy=0,
    heavy_k=1.5,
):
    """InferenceData with a log_likelihood group shaped for az.loo/az.compare.

    Per-observation log-likelihood centred at ``mean_ll`` with small posterior
    spread — enough draws for PSIS-LOO to run. The first ``n_exact``
    observations get a log-likelihood that is *identical across every draw*
    (what a clipped, saturated ``p_left`` produces; PSIS has no tail to fit
    there and arviz reports Pareto k = inf). The next ``n_heavy`` observations
    get genuinely heavy-tailed importance weights: log-lik = mean_ll -
    heavy_k * Exponential(1), whose weights exp(-loglik) follow a Pareto tail
    with GPD shape ≈ ``heavy_k`` (> 0.7 ⇒ PSIS genuinely unreliable there).
    """
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(seed)
    # (chain, draw, obs); slight per-draw jitter so importance sampling is stable.
    arr = mean_ll + rng.normal(0.0, 0.02, size=(n_chains, n_draws, n_obs))
    arr[:, :, :n_exact] = mean_ll
    arr[:, :, n_exact : n_exact + n_heavy] = mean_ll - heavy_k * rng.exponential(
        1.0, size=(n_chains, n_draws, n_heavy)
    )
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


def _patch_fits(monkeypatch, idata_by_name):
    """Make ``fit_models_cached`` hand back real FittedModel objects (no MCMC)."""
    from src.models import pymc_inference as pi

    def _fake_fit_models_cached(
        model_names, models_dir, responses_path, cache_dir=None, **kw
    ):
        return {
            m: pi.FittedModel(
                name=m, model=object(), idata=idata_by_name[m], fingerprint="fp"
            )
            for m in model_names
        }

    monkeypatch.setattr(
        "src.models.pymc_inference.fit_models_cached", _fake_fit_models_cached
    )


def test_compare_table_reports_elpd_diff_and_dse(tmp_path, monkeypatch):
    models_dir = _seed_models(tmp_path, ["good", "bad"])
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,x\n" + "1,0\n" * 25, encoding="utf-8")

    # "good" assigns higher per-obs log-likelihood than "bad".
    idata = {
        "good": _fake_idata(-0.3, seed=1),
        "bad": _fake_idata(-0.9, seed=2),
    }

    _patch_fits(monkeypatch, idata)

    table = mp.compare_table(responses, models_dir)

    assert set(table) == {"good", "bad"}
    # Best model has zero elpd_diff and zero dse against itself.
    assert table["good"]["elpd_diff"] == pytest.approx(0.0, abs=1e-9)
    assert table["good"]["dse"] == pytest.approx(0.0, abs=1e-9)
    # The worse model is behind (positive elpd_diff) with a real SE.
    assert table["bad"]["elpd_diff"] > 0
    assert table["bad"]["dse"] > 0
    assert "rank" in table["good"] and table["good"]["rank"] == 0


def test_compare_table_does_not_flag_exact_loo_points_as_unreliable(
    tmp_path, monkeypatch
):
    """A trial whose log-likelihood is identical across every posterior draw (a
    clipped, saturated ``p_left``) has no importance-weight tail for PSIS to fit:
    arviz reports Pareto k = inf and sets its blanket ``warning`` — but the LOO
    term for that trial is *exact* (all importance weights are equal). Such
    trials must not make the model "unreliable": that verdict was excluding
    genuine winners from export and zeroing their design prior."""
    models_dir = _seed_models(tmp_path, ["saturating", "smooth"])
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,x\n" + "1,0\n" * 25, encoding="utf-8")

    idata = {
        # 5 of 25 trials saturated: arviz's own loo(...).warning is True here.
        "saturating": _fake_idata(-0.3, seed=1, n_exact=5),
        "smooth": _fake_idata(-0.9, seed=2),
    }
    _patch_fits(monkeypatch, idata)

    table = mp.compare_table(responses, models_dir)

    assert table["saturating"]["loo_unreliable"] is False
    assert table["saturating"]["n_exact_loo_points"] == 5
    assert table["saturating"]["n_bad_k"] == 0
    assert table["saturating"]["frac_bad_k"] == pytest.approx(0.0)
    assert table["smooth"]["loo_unreliable"] is False
    assert table["smooth"]["n_exact_loo_points"] == 0
    # The saturating model still ranks first: its ELPD is used, not discarded.
    assert table["saturating"]["rank"] == 0


def test_compare_table_flags_genuinely_heavy_tailed_loo(tmp_path, monkeypatch):
    """When a real proportion of trials has heavy-tailed importance weights
    (finite Pareto k above arviz's good_k), PSIS-LOO *is* untrustworthy and the
    row must say so, with the proportion recorded for audit."""
    models_dir = _seed_models(tmp_path, ["heavy", "smooth"])
    responses = tmp_path / "responses.csv"
    responses.write_text("chose_left,x\n" + "1,0\n" * 25, encoding="utf-8")

    idata = {
        # 5 of 25 trials (20 %) with Pareto-tailed weights, far above tolerance.
        "heavy": _fake_idata(-0.3, seed=3, n_heavy=5),
        "smooth": _fake_idata(-0.9, seed=4),
    }
    _patch_fits(monkeypatch, idata)

    table = mp.compare_table(responses, models_dir)

    assert table["heavy"]["loo_unreliable"] is True
    assert table["heavy"]["n_bad_k"] == 5
    assert table["heavy"]["frac_bad_k"] == pytest.approx(0.2)
    assert np.isfinite(table["heavy"]["max_pareto_k"])
    assert table["heavy"]["max_pareto_k"] > 0.7
    assert table["smooth"]["loo_unreliable"] is False
