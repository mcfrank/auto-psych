"""MCMC convergence gates admission, export and pruning.

Divergences and poor R-hat used to print a warning and nothing else: a
non-converged fit could be admitted, prune rivals and be exported as the
discovered model. PSIS-LOO reliability (Pareto k) does not catch this: it
measures whether single trials dominate the fit, not whether the chains mixed.

A fit is converged when at most 0.1% of its transitions diverged and R-hat <=
1.05 and bulk ESS >= 100 on every free parameter (user decision 2026-09-26: the
stricter 1.01 / 400 / zero-divergence gate rejected most seeds at the sweep's
target_accept). A fit that fails as a near miss is refit once at
target_accept 0.95, with a random seed of its own, and that fit is used
everywhere. A model file's declared target_accept is a floor on
the loop's.
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


def test_more_than_a_tenth_of_a_percent_divergent_transitions_fail():
    rng = np.random.default_rng(0)
    diverging = np.zeros((4, 1000), dtype=bool)
    diverging[1, :10] = True  # 0.25% of 4000
    problems = convergence_problems(_idata(_iid(rng), diverging=diverging), ["theta"])
    assert any("10 divergent" in p for p in problems)


def test_a_few_divergent_transitions_are_tolerated():
    rng = np.random.default_rng(0)
    diverging = np.zeros((4, 1000), dtype=bool)
    diverging[1, :3] = True  # 0.075% of 4000
    assert convergence_problems(_idata(_iid(rng), diverging=diverging), ["theta"]) == []


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
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    monkeypatch.setattr(model_zoo, "fit_model", lambda *a, **k: object())
    monkeypatch.setattr(model_zoo, "convergence_problems_of", lambda fitted: ["12 divergent transitions"])
    monkeypatch.setattr(model_zoo, "convergence_diagnostics_of", lambda fitted: NEAR_MISS)
    verdict = model_zoo._admit_candidate_with_reason(
        cand / "candidate.py", models_dir, "iter0_candidate0", tmp_path / "r.csv",
        fit_kwargs={"target_accept": 0.8},  # the faithful sweep's setting
    )
    assert not verdict.admitted
    assert "did not converge" in verdict.reason and "12 divergent" in verdict.reason
    # The fitter already refit it with smaller steps: advice to declare
    # target_accept 0.95 made the repair re-sample the same failing fit.
    assert "even at target_accept 0.95" in verdict.reason
    assert "raising target_accept will not help" in verdict.reason
    assert "SAMPLER_SETTINGS" not in verdict.reason
    assert "non-centred" in verdict.reason and "weakly identified" in verdict.reason

    # A fit far from converging was not refit; the reason says why.
    monkeypatch.setattr(model_zoo, "convergence_diagnostics_of", lambda fitted: HOPELESS)
    verdict = model_zoo._admit_candidate_with_reason(
        cand / "candidate.py", models_dir, "iter0_candidate1", tmp_path / "r.csv",
        fit_kwargs={"target_accept": 0.8},
    )
    assert "too far from converging for smaller NUTS steps to help" in verdict.reason
    assert "R-hat <= 1.2" in verdict.reason and "even at" not in verdict.reason
    assert "raising target_accept will not help" in verdict.reason


def test_the_brief_does_not_offer_smaller_steps_as_a_convergence_fix(tmp_path):
    from src.pipelines.inner_loop.candidate_agent import _write_candidate_context
    from tests.inner_loop_fixtures import write_responses, write_seed_models

    _write_candidate_context(
        tmp_path / "candidate_0", write_responses(tmp_path), write_seed_models(tmp_path),
        iteration=0, candidate_idx=0, candidate_count=1, current_posterior=None,
    )
    context = (tmp_path / "candidate_0" / "CONTEXT.md").read_text(encoding="utf-8")
    assert "SAMPLER_SETTINGS" not in context
    assert "already refit once with smaller NUTS steps" in context
    assert "non-centred" in context


def _row(rank, *, unreliable=False, not_converged=False):
    return {"rank": rank, "elpd_loo": -100.0 - rank, "elpd_diff": float(rank),
            "dse": 1.0, "loo_unreliable": unreliable, "not_converged": not_converged}


def test_a_non_converged_model_is_never_exported():
    comparison = {"best_but_bad": _row(0, not_converged=True), "second": _row(1)}
    posterior = {"posteriors": {"best_but_bad": 0.6, "second": 0.4}}
    assert _best_exportable_model(posterior, comparison) == "second"


# A failed fit that smaller steps can plausibly fix, and one they cannot.
NEAR_MISS = pi.ConvergenceDiagnostics(n_divergent=25, n_draws=4000, max_r_hat=1.06, min_bulk_ess=80)
HOPELESS = pi.ConvergenceDiagnostics(n_divergent=1000, n_draws=4000, max_r_hat=2.5, min_bulk_ess=5)


class _StubFit(dict):
    """A stubbed fit: its settings, and a fingerprint made from them."""

    @property
    def fingerprint(self):
        return f"fp-{self['target_accept']}-{self['random_seed']}"


def _escalation(monkeypatch, *, converges_at, diagnostics=NEAR_MISS, seeds=None):
    """fit_model with the sampling stubbed: record each target_accept tried
    (and, in ``seeds``, each random seed)."""
    tried = []

    def fake_fit_once(name, models_dir, responses_path, settings, cache_dir):
        tried.append(settings["target_accept"])
        if seeds is not None:
            seeds.append(settings["random_seed"])
        return _StubFit(target_accept=settings["target_accept"], random_seed=settings["random_seed"])

    monkeypatch.setattr(pi, "_fit_once", fake_fit_once)
    monkeypatch.setattr(
        pi, "convergence_problems_of",
        lambda fitted: [] if fitted["target_accept"] >= converges_at else ["R-hat 1.2"],
    )
    monkeypatch.setattr(pi, "convergence_diagnostics_of", lambda fitted: diagnostics)
    monkeypatch.setattr(pi, "model_sampler_settings", lambda name, d: {})
    return tried


def test_a_near_miss_is_refit_once_at_0_95(tmp_path, monkeypatch):
    tried = _escalation(monkeypatch, converges_at=0.95)
    fitted = pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=4)
    assert tried == [0.8, 0.95]
    assert fitted["target_accept"] == 0.95


def test_the_refit_samples_with_a_random_seed_of_its_own(tmp_path, monkeypatch):
    """It used to reuse the first fit's seed (42), re-sampling from the same
    draws. Its seed now comes from the first fit's seed and fingerprint."""
    seeds = []
    _escalation(monkeypatch, converges_at=0.95, seeds=seeds)
    pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=4)
    first_fingerprint = _StubFit(target_accept=0.8, random_seed=42).fingerprint
    assert seeds == [42, pi.refit_random_seed(42, first_fingerprint)]
    assert seeds[1] != 42


def test_refit_seeds_are_deterministic_and_differ_between_fits():
    assert pi.refit_random_seed(42, "aaaa") == pi.refit_random_seed(42, "aaaa")
    seeds = {pi.refit_random_seed(42, fp) for fp in ("aaaa", "bbbb", "cccc")}
    assert len(seeds) == 3 and 42 not in seeds
    assert pi.refit_random_seed(7, "aaaa") != pi.refit_random_seed(42, "aaaa")
    assert all(0 <= seed < 2**31 for seed in seeds)


def test_the_refit_seed_is_part_of_the_refits_cache_fingerprint(tmp_path):
    """A resume finds the refit it sampled, and never a refit sampled at the
    first fit's seed (the old code's) in its place."""
    (tmp_path / "m.py").write_text("# model\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("# another model\n", encoding="utf-8")
    responses = tmp_path / "r.csv"
    responses.write_text("chose_left\n1\n", encoding="utf-8")
    settings = pi.resolve_fit_settings("m", tmp_path, {"target_accept": 0.8})
    first = pi.fit_fingerprint("m", tmp_path, responses, settings)
    refit = pi.refit_settings(settings, first)
    assert refit["target_accept"] == 0.95
    assert refit["random_seed"] == pi.refit_random_seed(settings["random_seed"], first)
    old_style = {**settings, "target_accept": 0.95}
    assert pi.fit_fingerprint("m", tmp_path, responses, refit) != pi.fit_fingerprint(
        "m", tmp_path, responses, old_style
    )
    other_first = pi.fit_fingerprint("other", tmp_path, responses, settings)
    assert pi.refit_settings(settings, other_first)["random_seed"] != refit["random_seed"]


def test_a_fit_far_from_converging_is_not_refit(tmp_path, monkeypatch, capsys):
    """A chain stuck in another mode (R-hat 2.5, ESS 5) is a geometry problem:
    the refit at 0.95 used to double the cost of the same rejection."""
    tried = _escalation(monkeypatch, converges_at=0.95, diagnostics=HOPELESS)
    fitted = pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=4)
    assert tried == [0.8]
    assert fitted["target_accept"] == 0.8  # returned as is; the gate rejects it
    assert "not refitting" in capsys.readouterr().out


@pytest.mark.parametrize(
    "diagnostics, near",
    [
        (NEAR_MISS, True),
        (pi.ConvergenceDiagnostics(80, 4000, 1.2, 20), True),  # every threshold met exactly
        (pi.ConvergenceDiagnostics(81, 4000, 1.05, 400), False),  # > 2% divergent
        (pi.ConvergenceDiagnostics(0, 4000, 1.21, 400), False),  # R-hat above 1.2
        (pi.ConvergenceDiagnostics(0, 4000, 1.03, 19), False),  # bulk ESS below 20
        (pi.ConvergenceDiagnostics(None, 4000, 1.03, 400), False),  # no divergence statistic
        (pi.ConvergenceDiagnostics(0, 4000, float("nan"), 400), False),  # R-hat undefined
        (HOPELESS, False),
    ],
)
def test_a_near_miss_is_close_on_every_diagnostic(diagnostics, near):
    assert pi.is_near_miss(diagnostics) is near


def test_a_converged_fit_is_not_refit(tmp_path, monkeypatch):
    tried = _escalation(monkeypatch, converges_at=0.8)
    pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=4)
    assert tried == [0.8]


def test_the_refit_is_not_repeated_if_it_also_fails(tmp_path, monkeypatch):
    tried = _escalation(monkeypatch, converges_at=1.1)
    fitted = pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=4)
    assert tried == [0.8, 0.95]
    assert fitted["target_accept"] == 0.95  # the gate then rejects it


def test_a_single_chain_smoke_fit_is_never_refit(tmp_path, monkeypatch):
    """With one chain R-hat is undefined; the candidate self-check's smoke fit
    would always 'fail'."""
    tried = _escalation(monkeypatch, converges_at=1.1)
    pi.fit_model("m", tmp_path, tmp_path / "r.csv", target_accept=0.8, chains=1)
    assert tried == [0.8]
