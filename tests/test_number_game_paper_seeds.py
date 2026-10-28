"""The Bigelow & Piantadosi (2016) seeds: the paper's Independent and Mixture models.

Their concepts are LOTlib's `independent_grammar` (the paper's implementation)
evaluated over 1..100, with three of its slips corrected to the paper's
wording: "odd numbers" include 1 (the grammar's 2x + 1 starts at 3), primes
exclude 1 (`isprime_(1)` returns 1), and the Mixture Model has the Independent
model's 45 rules (LOTlib's `mix_grammar` keeps multiples of 10, which the
paper says the two share). An interval [m, n] is drawn as two constants of
1..100, so each has prior (1 - lambda_0) / 100^2; the m > n draws are empty
sets, which have no prior.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pymc as pm
import pytest

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.runtime.config import REPO_ROOT

SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
SEEDS = ["bigelow_piantadosi_independent", "bigelow_piantadosi_mixture"]
N = np.arange(1, 101)


def _module(seed):
    spec = importlib.util.spec_from_file_location(seed, SEEDS_DIR / f"{seed}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_rules():
    rules = {
        "odd": list(range(1, 100, 2)),
        "primes": [n for n in N if n > 1 and all(n % a for a in range(2, n))],
        "squares": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "cubes": [1, 8, 27, 64],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {i}": list(range(i, 101, i)) for i in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {i}": [i**k for k in range(1, 8) if i**k <= 100] for i in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in N if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in N if str(d) in str(n)] for d in range(10)})
    return rules


@pytest.mark.parametrize("seed", SEEDS)
def test_the_rules_are_lotlibs_independent_grammar(seed):
    module = _module(seed)
    assert dict(zip(module.RULE_NAMES, map(sorted, module.RULES))) == {
        name: sorted(map(int, s)) for name, s in _expected_rules().items()
    }


def _brute_force_p_yes(number_set, lambda_0, rule_weights):
    """P(target in C | D) by summing over all 45 rules and 5,050 intervals."""
    rules = list(_expected_rules().values())
    intervals = [list(range(m, n + 1)) for m in N for n in range(m, 101)]
    member = np.array([np.isin(N, c) for c in rules + intervals], dtype=float)
    log_lik = np.log(0.9 * member / member.sum(1, keepdims=True) + 0.1 / 100)[:, np.array(number_set) - 1].sum(1)
    log_prior = np.concatenate([np.log(lambda_0 * rule_weights), np.full(len(intervals), np.log((1 - lambda_0) / 100**2))])
    post = np.exp(log_prior + log_lik - np.logaddexp.reduce(log_prior + log_lik))
    return post @ member


@pytest.mark.parametrize("seed", SEEDS)
def test_p_yes_is_the_posterior_over_every_rule_and_interval(seed):
    sets = [(16, 8), (60, 80, 10, 30), (3,), (81, 25, 4, 36)]
    rows = [{"number_set": ",".join(map(str, s)), "target": t, "rating": 0} for s in sets for t in N]
    model = load_pymc_model(seed, SEEDS_DIR)
    with model:
        pm.set_data(make_stim_data(model, rows))
    rng = np.random.default_rng(0)
    lambda_0, weights = 0.3, rng.dirichlet(np.full(45, 2.0))
    point = {"lambda_0": lambda_0}
    if seed == "bigelow_piantadosi_independent":
        point.update(_free_weights(_module(seed), weights))
    else:
        weights = np.full(45, 1 / 45)
    values = {
        model.rvs_to_values[rv].name: model.rvs_to_transforms[rv].forward(point[rv.name], *rv.owner.inputs).eval()
        for rv in model.free_RVs
    }
    [p_yes_graph] = model.replace_rvs_by_values([model["p_yes"]])
    p_yes = model.compile_fn(p_yes_graph, inputs=model.value_vars, on_unused_input="ignore")(values)
    expected = np.concatenate([_brute_force_p_yes(s, lambda_0, weights) for s in sets])
    np.testing.assert_allclose(p_yes, expected, rtol=1e-9, atol=1e-12)


def _free_weights(module, rule_weights):
    """The Independent seed's free weights for the given 45 rule weights."""
    names = module.RULE_NAMES
    end, contains = names.index("ends in 0"), names.index("contains digit 0")
    set_weights = np.delete(rule_weights, contains)
    set_weights[end if end < contains else end - 1] = rule_weights[end] + rule_weights[contains]
    return {"set_weights": set_weights, "zero_split": rule_weights[end] / (rule_weights[end] + rule_weights[contains])}


def test_the_priors_are_the_papers():
    """Gamma(2, 1) on each unnormalised production weight: normalised, the rule
    weights are Dirichlet(2, ..., 2) and lambda_0 = a / (a + b) is Beta(2, 2)."""
    for seed in SEEDS:
        model = load_pymc_model(seed, SEEDS_DIR)
        assert model["lambda_0"].owner.op.name == "beta"
        assert [float(x.eval()) for x in model["lambda_0"].owner.inputs[-2:]] == [2.0, 2.0]
    assert "rule_weights" not in load_pymc_model("bigelow_piantadosi_mixture", SEEDS_DIR).named_vars
    # "ends in 0" and "contains digit 0" are one set, so the data fix only their
    # sum: the seed samples the 44 distinct sets' weights and the split apart,
    # which is the same prior (Dirichlet aggregation: the pair's sum is a
    # concentration-4 component, and its split an independent Beta(2, 2)).
    independent = load_pymc_model("bigelow_piantadosi_independent", SEEDS_DIR)
    draws = pm.draw(independent["rule_weights"], draws=40_000, random_seed=1)
    reference = np.random.default_rng(2).dirichlet(np.full(45, 2.0), size=40_000)
    np.testing.assert_allclose(draws.mean(0), reference.mean(0), atol=0.001)
    np.testing.assert_allclose(draws.std(0), reference.std(0), atol=0.001)
    names = _module("bigelow_piantadosi_independent").RULE_NAMES
    end, contains = names.index("ends in 0"), names.index("contains digit 0")
    assert np.corrcoef(draws[:, end], draws[:, contains])[0, 1] == pytest.approx(
        np.corrcoef(reference[:, end], reference[:, contains])[0, 1], abs=0.02
    )
