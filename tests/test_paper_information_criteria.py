"""AIC and BIC beside ELPD (``scripts/paper/information_criteria.py``): the
maximized likelihood, the parameter counts under both conventions, and the
marginal likelihood over per-person effects, checked on toy models whose
answers are known in closed form or by quadrature."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, optimize, special

from scripts.paper import information_criteria as ic

GROUPED = '''
import numpy as np
import pymc as pm

with pm.Model() as model:
    group = pm.Data("group", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    theta = pm.Uniform("theta", lower=0.0, upper=1.0, shape=2)
    pm.Bernoulli("response", p=theta[group], observed=chose_left)
'''

REDUNDANT = '''
import numpy as np
import pymc as pm

with pm.Model() as model:
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    a = pm.Normal("a", 0.0, 1.0)
    b = pm.Normal("b", 0.0, 1.0)
    pm.Bernoulli("response", p=pm.math.sigmoid(a + b), observed=chose_left)
'''

# Random intercepts, non-centered, five slots of which the data use three.
HIERARCHICAL = '''
import numpy as np
import pymc as pm

with pm.Model() as model:
    slot = pm.Data("slot", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    mu = pm.Normal("mu", 0.0, 2.0)
    sigma = pm.HalfNormal("sigma", 2.0)
    z = pm.Normal("z", 0.0, 1.0, shape=5)
    pm.Bernoulli("response", p=pm.math.sigmoid(mu + sigma * z[slot]), observed=chose_left)
'''

# Per-person successes out of 12 trials; slots 0, 1, 3 used (slot 2 and 4 empty).
PEOPLE = {"p_a": (0, 3), "p_b": (1, 7), "p_c": (3, 10)}
N_PER = 12


def _write(tmp_path: Path, name: str, source: str, rows) -> tuple[Path, Path]:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / f"{name}.py").write_text(source)
    responses = tmp_path / f"{name}.csv"
    with responses.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return models, responses


def _bern_ll(k: int, n: int, p: float) -> float:
    return k * math.log(p) + (n - k) * math.log(1 - p)


def _hier_rows():
    rows = []
    for pid, (slot, k) in PEOPLE.items():
        rows += [{"participant_id": pid, "slot": slot, "chose_left": int(t < k)} for t in range(N_PER)]
    return rows


def test_aic_and_bic_follow_their_definitions():
    out = ic.criteria(log_lik=-100.0, k=3, n=50)
    assert out == pytest.approx({"aic": 206.0, "bic": 200.0 + 3 * math.log(50)})


def test_the_mle_of_a_grouped_bernoulli_is_its_group_means(tmp_path):
    rows = [{"group": g, "chose_left": int(t < k)} for g, k in ((0, 3), (1, 8)) for t in range(10)]
    models, responses = _write(tmp_path, "grouped", GROUPED, rows)
    lik = ic.PyMCLikelihood.build(models, "grouped", responses)
    fit = ic.maximize(lik, starts=[lik.x0])
    assert fit.log_lik == pytest.approx(_bern_ll(3, 10, 0.3) + _bern_ll(8, 10, 0.8), abs=1e-5)
    # The climb is on record: the log-likelihood every so many iterations, ending at the maximum.
    values = [v for _, v in fit.trajectory]
    assert fit.iterations >= 1 and values == sorted(values) and values[-1] == pytest.approx(fit.log_lik)


def test_the_climb_is_logged_along_the_way(tmp_path, monkeypatch):
    rows = [{"group": g, "chose_left": int(t < k)} for g, k in ((0, 3), (1, 8)) for t in range(10)]
    models, responses = _write(tmp_path, "grouped", GROUPED, rows)
    lik = ic.PyMCLikelihood.build(models, "grouped", responses)
    monkeypatch.setattr(ic, "TRAJECTORY_EVERY", 1)
    lines = []
    fit = ic.maximize(lik, starts=[lik.x0], log=lines.append)
    values = [v for _, v in fit.trajectory]
    assert len(values) == fit.iterations + 1 and values == sorted(values)
    assert sum(line.startswith("iteration") for line in lines) == fit.iterations


def test_coordinates_split_into_per_person_population_and_unused(tmp_path):
    models, responses = _write(tmp_path, "hier", HIERARCHICAL, _hier_rows())
    lik = ic.PyMCLikelihood.build(models, "hier", responses)
    coords = ic.classify_coordinates(lik, lik.x0, units=ic.unit_column(responses, "participant_id"))
    names = lik.coordinate_names
    assert sorted(names[i] for i in coords.population) == ["mu", "sigma_log__"]
    assert {u: [names[i] for i in idx] for u, idx in coords.per_unit.items()} == {
        "p_a": ["z[0]"], "p_b": ["z[1]"], "p_c": ["z[3]"]}
    assert sorted(names[i] for i in coords.unused) == ["z[2]", "z[4]"]
    assert coords.k_conditional == 5 and coords.k_marginal == 2


def test_the_jacobian_rank_drops_a_redundant_parameter(tmp_path):
    rows = [{"chose_left": int(t < 4)} for t in range(10)]
    models, responses = _write(tmp_path, "redundant", REDUNDANT, rows)
    lik = ic.PyMCLikelihood.build(models, "redundant", responses)
    coords = ic.classify_coordinates(lik, lik.x0, units=None)
    assert coords.k_conditional == 2
    assert ic.jacobian_rank(lik.pointwise, lik.x0 + 0.3, coords.active) == 1


def test_the_conditional_mle_fits_each_person_their_own_rate(tmp_path):
    models, responses = _write(tmp_path, "hier", HIERARCHICAL, _hier_rows())
    lik = ic.PyMCLikelihood.build(models, "hier", responses)
    fit = ic.maximize(lik, starts=[lik.x0])
    expected = sum(_bern_ll(k, N_PER, k / N_PER) for _, k in PEOPLE.values())
    assert fit.log_lik == pytest.approx(expected, abs=1e-4)


def _quadrature_marginal(mu: float, sigma: float) -> float:
    total = 0.0
    for _, k in PEOPLE.values():
        def integrand(z):
            p = special.expit(mu + sigma * z)
            return math.exp(_bern_ll(k, N_PER, p)) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        total += math.log(integrate.quad(integrand, -12, 12, epsabs=1e-14)[0])
    return total


def test_the_marginal_likelihood_matches_quadrature_at_its_maximum(tmp_path):
    models, responses = _write(tmp_path, "hier", HIERARCHICAL, _hier_rows())
    lik = ic.PyMCLikelihood.build(models, "hier", responses)
    units = ic.unit_column(responses, "participant_id")
    coords = ic.classify_coordinates(lik, lik.x0, units=units)
    out = ic.marginal_log_lik(lik, coords, units, start=lik.x0, n_samples=4000, seed=0)
    best = optimize.minimize(lambda v: -_quadrature_marginal(v[0], math.exp(v[1])), [0.0, 0.0],
                             method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9})
    assert out.log_lik == pytest.approx(-best.fun, abs=0.02)
    assert out.log_lik <= -best.fun + 0.02
    assert out.laplace == pytest.approx(-best.fun, abs=0.2)
    assert out.min_ess > 1000


def test_a_model_record_carries_both_conventions(tmp_path):
    models, responses = _write(tmp_path, "hier", HIERARCHICAL, _hier_rows())
    record = ic.pymc_record(models, "hier", responses, unit="participant_id", starts=[], n_samples=2000)
    n = 3 * N_PER
    assert record["n"] == n
    assert record["conditional"]["k"] == 5 and record["marginal"]["k"] == 2
    assert record["conditional"]["bic"] == pytest.approx(
        -2 * record["conditional"]["log_lik"] + 5 * math.log(n))
    assert record["marginal"]["log_lik"] < record["conditional"]["log_lik"]
    # The evidence averages the marginal likelihood over the population prior: below its maximum.
    assert record["evidence"]["log_evidence"] < record["marginal"]["log_lik"]


def test_a_flat_model_record_has_one_convention(tmp_path):
    rows = [{"group": g, "chose_left": int(t < k)} for g, k in ((0, 3), (1, 8)) for t in range(10)]
    models, responses = _write(tmp_path, "grouped", GROUPED, rows)
    record = ic.pymc_record(models, "grouped", responses, unit=None, starts=[], n_samples=4000)
    assert record["conditional"]["k"] == 2 and record["marginal"] is None
    assert record["evidence"]["log_evidence"] == pytest.approx(
        special.betaln(4, 8) + special.betaln(9, 3), abs=0.01)


def test_the_tasks_are_each_runs_winner_and_seeds_and_every_rsa_final_and_seed():
    tasks = ic.tasks()
    by = {}
    for t in tasks:
        by.setdefault((t.domain, t.run), []).append(t)
        assert t.file.exists(), t
    # Subjective randomness and the number game: every run's winner and the seeds on all runs pooled.
    assert set(by) == {("subjective_randomness", "pooled"), ("number_game", "pooled"), ("rsa", "pooled")}
    sr = {t.model: t.role for t in by[("subjective_randomness", "pooled")]}
    assert sorted(sr.values()) == ["seed"] * 4 + ["winner"] * 3
    assert {"local_representativeness", "motif_stack", "falk_konold_dp",
            "finite_experience_occurrence"} <= set(sr)
    ng = {t.model: t.role for t in by[("number_game", "pooled")]}
    assert sorted(ng.values()) == ["seed"] * 3 + ["winner"] * 3
    for t in by[("subjective_randomness", "pooled")] + by[("number_game", "pooled")]:
        assert t.responses is None and len(t.sources) == 3 and all(p.exists() for p in t.sources)
    rsa = {t.model: t.role for t in by[("rsa", "pooled")]}
    assert {m for m, r in rsa.items() if r == "winner"} == {
        "boundary_avoidable_ambiguity_listener", "feature_surprisal_ensemble_listener",
        "competition_endpoint_binding_lapse_l2"}
    assert {"rsa_l1", "rsa_l2", "rsa_l2_singleton_distinct_feat_color_l0"} <= {
        m for m, r in rsa.items() if r == "seed"}
    assert len(rsa) == 15


def test_a_comparison_is_in_nats_and_positive_when_the_winner_is_preferred():
    winner = {"n": 100, "conditional": {"log_lik": -50.0, "k": 10, **ic.criteria(-50.0, 10, 100)},
              "marginal": {"log_lik": -60.0, "k": 3, **ic.criteria(-60.0, 3, 100)},
              "evidence": {"log_evidence": -75.0}}
    seed = {"n": 100, "conditional": {"log_lik": -70.0, "k": 2, **ic.criteria(-70.0, 2, 100)}, "marginal": None,
            "evidence": {"log_evidence": -78.5}}
    out = ic.compare(winner, seed)
    assert out["log_bf"] == pytest.approx(3.5)
    assert out["conditional"] == pytest.approx(
        {"log_lik": 20.0, "k": 8, "aic": 12.0, "bic": 20.0 - 4 * math.log(100)})
    # A seed without per-person coordinates is the same under both conventions.
    assert out["marginal"] == pytest.approx(
        {"log_lik": 10.0, "k": 1, "aic": 9.0, "bic": 10.0 - 0.5 * math.log(100)})


CATEGORICAL = '''
import numpy as np
import pymc as pm

with pm.Model() as model:
    choice = pm.Data("choice", np.zeros(1, dtype="int64"))
    w = pm.Dirichlet("w", a=np.array([1.0, 2.0, 3.0]))
    pm.Categorical("response", p=w, observed=choice)
'''


def test_the_evidence_of_a_flat_model_is_its_beta_bernoulli_integral(tmp_path):
    rows = [{"group": g, "chose_left": int(t < k)} for g, k in ((0, 3), (1, 8)) for t in range(10)]
    models, responses = _write(tmp_path, "grouped", GROUPED, rows)
    lik = ic.PyMCLikelihood.build(models, "grouped", responses)
    ev = ic.flat_log_evidence(lik, start=lik.x0, n_samples=4000, seed=0)
    exact = special.betaln(4, 8) + special.betaln(9, 3)  # uniform priors: B(k + 1, n - k + 1) per group
    assert ev.log_evidence == pytest.approx(exact, abs=0.01)
    assert ev.laplace == pytest.approx(exact, abs=0.3)
    assert ev.min_ess > 1000


def test_the_evidence_counts_a_multivariate_prior(tmp_path):
    counts = (2, 5, 9)
    rows = [{"choice": c} for c, n in enumerate(counts) for _ in range(n)]
    models, responses = _write(tmp_path, "categorical", CATEGORICAL, rows)
    lik = ic.PyMCLikelihood.build(models, "categorical", responses)
    ev = ic.flat_log_evidence(lik, start=lik.x0, n_samples=4000, seed=0)
    alpha = np.array([1.0, 2.0, 3.0])
    exact = (special.gammaln(alpha.sum()) - special.gammaln(alpha.sum() + sum(counts))
             + sum(special.gammaln(a + n) - special.gammaln(a) for a, n in zip(alpha, counts)))
    assert ev.log_evidence == pytest.approx(exact, abs=0.01)


def _quadrature_evidence() -> float:
    """log of the integral over mu ~ N(0, 2), sigma ~ HalfNormal(2) and each person's z ~ N(0, 1)."""
    from numpy.polynomial.hermite_e import hermegauss

    nodes, weights = hermegauss(80)
    weights = weights / math.sqrt(2 * math.pi)
    mu = np.linspace(-12, 12, 1201)[:, None, None]
    sigma = np.linspace(1e-6, 12, 1201)[None, :, None]
    log_lik = 0.0
    for _, k in PEOPLE.values():
        p = special.expit(mu + sigma * nodes)
        log_lik = log_lik + np.log((np.exp(k * np.log(p) + (N_PER - k) * np.log1p(-p)) * weights).sum(-1))
    log_prior = (-0.5 * (mu[..., 0] / 2) ** 2 - math.log(2 * math.sqrt(2 * math.pi))
                 + math.log(2) - 0.5 * (sigma[..., 0] / 2) ** 2 - math.log(2 * math.sqrt(2 * math.pi)))
    integrand = np.exp(log_lik + log_prior)
    return math.log(integrate.trapezoid(integrate.trapezoid(integrand, sigma[0, :, 0], axis=1), mu[:, 0, 0]))


def test_the_evidence_of_a_hierarchical_model_matches_quadrature(tmp_path):
    models, responses = _write(tmp_path, "hier", HIERARCHICAL, _hier_rows())
    lik = ic.PyMCLikelihood.build(models, "hier", responses)
    units = ic.unit_column(responses, "participant_id")
    coords = ic.classify_coordinates(lik, lik.x0, units=units)
    ev = ic.hierarchical_log_evidence(lik, coords, units, start=lik.x0, n_outer=1000, n_inner=1000, seed=0)
    assert ev.log_evidence == pytest.approx(_quadrature_evidence(), abs=0.05)
    assert ev.min_ess > 200


def test_the_rsa_evidence_integrates_the_lapse_prior():
    from pyprojroot import here
    from src.rsa.context import Context
    from src.rsa.model_file import RSAModel

    path = here() / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "seed_models" / "literal_listener.py"
    ctx = Context(objects=((1, 0), (0, 1), (0, 0)), feature_names=("f0", "f1"), utterance=0)
    choices = [0] * 16 + [1] * 2 + [2] * 2
    lik = ic.RSALikelihood.build(RSAModel(path, name="literal_listener"), [ctx] * 20, choices)
    # A skewed one-dimensional posterior: 4,000 draws leave about +-0.01 of Monte Carlo noise.
    ev = ic.flat_log_evidence(lik, start=lik.x0, n_samples=20000, seed=0)
    # lapse ~ Beta(1, 9); P(target) = 1 - 2 lapse / 3, each other object lapse / 3.
    exact = math.log(integrate.quad(
        lambda l: (1 - 2 * l / 3) ** 16 * (l / 3) ** 4 * 9 * (1 - l) ** 8, 0, 1, epsabs=1e-16)[0])
    assert ev.log_evidence == pytest.approx(exact, abs=0.01)


def test_the_rsa_mle_recovers_a_literal_listeners_lapse():
    from pyprojroot import here
    from src.rsa.context import Context
    from src.rsa.model_file import RSAModel

    path = here() / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "seed_models" / "literal_listener.py"
    # Three objects, the word true of object 0 only; 16 of 20 choose it.
    ctx = Context(objects=((1, 0), (0, 1), (0, 0)), feature_names=("f0", "f1"), utterance=0)
    choices = [0] * 16 + [1] * 2 + [2] * 2
    lik = ic.RSALikelihood.build(RSAModel(path, name="literal_listener"), [ctx] * 20, choices)
    fit = ic.maximize(lik, starts=[lik.x0])
    # P(target) = (1 - lapse) + lapse / 3 = 0.8, each other object lapse / 3 = 0.1.
    assert fit.log_lik == pytest.approx(16 * math.log(0.8) + 4 * math.log(0.1), abs=1e-4)
    assert ic.jacobian_rank(lik.class_probs, fit.x, np.arange(lik.dim)) == 1


def test_the_pooled_responses_are_every_runs_final_data():
    tasks = {t.domain: t for t in ic.tasks() if t.run == "pooled" and t.domain != "rsa"}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for domain, n, people in (("subjective_randomness", 23104, 359), ("number_game", 22680, None)):
            rows = ic.unit_column_rows(ic.pooled_responses(tasks[domain], Path(tmp)))
            assert len(rows) == n
            if people:
                assert len({r["participant_id"] for r in rows}) == people


def test_the_pooled_elpd_beside_each_comparison_is_figure_2s():
    _, sr = ic._elpd_differences("subjective_randomness", "pooled", "personal_pattern_detection_gain")
    assert sr["falk_konold_dp"] == pytest.approx(3260.2 + 1146.0, abs=0.1)  # PSIS-LOO on all runs pooled
    _, ng = ic._elpd_differences("number_game", "pooled", "even_rule_digit_size_principle_counting")
    assert ng["bigelow_piantadosi_independent"] == pytest.approx(-9554.1 + 10666.6, abs=0.1)  # the unseen example sets
