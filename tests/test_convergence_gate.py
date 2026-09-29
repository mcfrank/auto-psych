"""MCMC convergence gates admission, export and pruning.

Divergences and poor R-hat used to print a warning and nothing else: a
non-converged fit could be admitted, prune rivals and be exported as the
discovered model. PSIS-LOO reliability (Pareto k) does not catch this: it
measures whether single trials dominate the fit, not whether the chains mixed.

A fit is converged when it has no divergent transitions, R-hat <= 1.01 and bulk
ESS >= 400 on every free parameter (Vehtari et al., 2021). A model file's
declared target_accept is a floor on the loop's, so a candidate rejected for
divergences can be fixed by declaring a higher one.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
from src.models import pymc_inference as pi
from src.models.pymc_inference import convergence_problems
from src.pipelines.inner_loop.scoring import _best_exportable_model


def _idata(chains, *, diverging=None):
    posterior = {"theta": np.asarray(chains, dtype=float)}
    shape = posterior["theta"].shape
    stats = {"diverging": np.zeros(shape, dtype=bool) if diverging is None else diverging}
    return az.from_dict(posterior=posterior, sample_stats=stats)


def _iid(rng, n_chains=4, n_draws=1000, offsets=(0, 0, 0, 0)):
    return np.stack([rng.normal(o, 1, n_draws) for o in offsets[:n_chains]])


def test_well_mixed_chains_pass():
    rng = np.random.default_rng(0)
    assert convergence_problems(_idata(_iid(rng)), ["theta"]) == []


def test_divergent_transitions_fail():
    rng = np.random.default_rng(0)
    diverging = np.zeros((4, 1000), dtype=bool)
    diverging[1, :3] = True
    problems = convergence_problems(_idata(_iid(rng), diverging=diverging), ["theta"])
    assert any("3 divergent" in p for p in problems)


def test_chains_that_disagree_fail_on_r_hat():
    rng = np.random.default_rng(0)
    problems = convergence_problems(_idata(_iid(rng, offsets=(0, 0, 0, 3))), ["theta"])
    assert any("R-hat" in p for p in problems)


def test_a_sticky_sampler_fails_on_effective_sample_size():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 0.02, size=(4, 1000)), axis=1)
    problems = convergence_problems(_idata(walk), ["theta"])
    assert any("ESS" in p for p in problems)


def test_a_trace_without_a_divergence_record_fails():
    rng = np.random.default_rng(0)
    idata = az.from_dict(posterior={"theta": _iid(rng)})
    assert any("divergence" in p for p in convergence_problems(idata, ["theta"]))


def test_a_declared_target_accept_is_a_floor_on_the_loops(tmp_path):
    (tmp_path / "careful.py").write_text(
        'SAMPLER_SETTINGS = {"target_accept": 0.95}\nmodel = None\n', encoding="utf-8"
    )
    assert pi.resolve_fit_settings("careful", tmp_path, {"target_accept": 0.8})["target_accept"] == 0.95
    assert pi.resolve_fit_settings("careful", tmp_path, {"target_accept": 0.99})["target_accept"] == 0.99


def test_a_non_converged_candidate_is_rejected_with_the_numbers_and_the_fix(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "People do X."}]}),
        encoding="utf-8",
    )
    cand = tmp_path / "candidate_0"
    cand.mkdir()
    (cand / "candidate.py").write_text("# candidate\n", encoding="utf-8")
    (cand / "hypothesis.md").write_text("People do Y.\n", encoding="utf-8")
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda n, d: object())
    monkeypatch.setattr(model_zoo, "model_logp_is_finite", lambda *a, **k: (True, ""))
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: ["12 divergent transitions"])
    verdict = model_zoo._admit_candidate_with_reason(
        cand / "candidate.py", models_dir, "iter0_candidate0", tmp_path / "r.csv"
    )
    assert not verdict.admitted
    assert "did not converge" in verdict.reason and "12 divergent" in verdict.reason
    assert "SAMPLER_SETTINGS" in verdict.reason


def _row(rank, *, unreliable=False, not_converged=False):
    return {"rank": rank, "elpd_loo": -100.0 - rank, "elpd_diff": float(rank),
            "dse": 1.0, "loo_unreliable": unreliable, "not_converged": not_converged}


def test_a_non_converged_model_is_never_exported():
    comparison = {"best_but_bad": _row(0, not_converged=True), "second": _row(1)}
    posterior = {"posteriors": {"best_but_bad": 0.6, "second": 0.4}}
    assert _best_exportable_model(posterior, comparison) == "second"
