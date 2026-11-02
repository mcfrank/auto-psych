"""Refinement of power_law_interval_endpoints: Bayesian concept learning over
45 rules and all intervals [m, n] of 1-100 (size principle, 10% of examples
uniform on 1-100, a prior weight per rule, a share lambda_0 for the rules as a
group), with interval endpoints drawn from a fitted power law (m n)^-gamma, and
the one change: the interval prior is also weighted by an Erlang density on
the interval's size s, s exp(-s / sigma), with sigma fitted (the component of
erlang_interval_prior_concepts), so people expect a range to have both a
location-dependent and a characteristic-width prior. "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# The 5,050 intervals [m, n], m <= n, as flat vectors.
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
SIZE = (HI - LO + 1).astype("float64")
LOG_MN = np.log(LO * HI.astype("float64"))
CONTAINS = ((LO[:, None] <= NUMBERS[None, :]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")


def interval_evidence(examples):
    """Per-interval likelihood of the examples (scaled) and its log scale."""
    ll = np.zeros(len(LO))
    for x in examples:
        inside = (LO <= x) & (x <= HI)
        ll += np.log(ALPHA * inside / SIZE + (1 - ALPHA) / 100)
    scale = ll.max()
    return np.exp(ll - scale), scale


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_ev = pm.Data("interval_ev", np.zeros((1, len(LO))))  # (sets, intervals)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Power-law endpoint exponent (0 = uniform endpoints) and Erlang typical size.
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    log_prior = -gamma * LOG_MN + np.log(SIZE) - SIZE / sigma
    prior = pt.exp(log_prior - pt.logsumexp(log_prior))  # (intervals,)
    weighted = interval_ev * prior[None, :]  # (S, intervals)
    total_mass = weighted.sum(axis=1)
    mass = weighted @ pt.as_tensor(CONTAINS)  # (S, 100): intervals containing t

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_interval = pt.log1p(-lambda_0) + pt.log(total_mass) + interval_log_scale
    log_joint = pt.concatenate([log_rules, log_interval[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - log_norm)
    post_interval = pt.exp(log_interval[:, None] - log_norm)
    in_concept = post_rules @ pt.as_tensor(RULE_MEMBERSHIP) + post_interval * mass / total_mass[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1.0 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ev = [interval_evidence(s) for s in unique]
    return {
        "rule_log_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_ev": np.stack([e for e, _ in ev]),
        "interval_log_scale": np.array([sc for _, sc in ev]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
