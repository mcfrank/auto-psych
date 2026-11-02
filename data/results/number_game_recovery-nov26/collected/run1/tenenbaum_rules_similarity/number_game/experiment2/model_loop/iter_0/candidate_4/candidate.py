"""Refinement of `erlang_interval_size_family_prior` with learned example
reliability.

People do the same Bayesian concept learning over 45 rules and the 5,050
intervals of 1..100: the size principle, a prior mass per concept family (the
exception rules in their own family) and an Erlang prior over interval width
with a learned characteristic size. The one change: the fraction of examples
people believe the program drew from its concept (rather than at random from
1..100) is not fixed at 90% but inferred. People who distrust the examples let
a concept that misses one of them survive, so they generalise from the
consistent majority and hesitate even on targets that were shown as examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
INTERVAL_WIDTH = CONCEPT_SIZE[N_RULES:]

with pm.Model() as model:
    # Per distinct set: how many examples each concept contains / misses.
    n_in = pm.Data("n_in", np.zeros((1, len(MEMBERSHIP))))  # (sets, concepts)
    n_out = pm.Data("n_out", np.zeros((1, len(MEMBERSHIP))))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Believed probability that an example came from the concept (was fixed at 0.9).
    reliability_logit = pm.Normal("reliability_logit", mu=2.2, sigma=1.5)
    reliability = pm.Deterministic("reliability", pm.math.sigmoid(reliability_logit))

    # Example likelihood under each concept (size principle plus random draws).
    log_member = pt.log(reliability / CONCEPT_SIZE + (1 - reliability) / 100.0)
    log_nonmember = pt.log((1 - reliability) / 100.0)
    set_log_lik = n_in * log_member[None, :] + n_out * log_nonmember

    erlang = pt.log(INTERVAL_WIDTH) - INTERVAL_WIDTH / interval_scale
    log_per_interval = erlang - pt.logsumexp(erlang)
    log_prior = pt.concatenate([(log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY], log_w[6] + log_per_interval])

    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_in = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_out = np.array([len(s) for s in unique], dtype="float64")[:, None] - n_in
    return {
        "n_in": n_in,
        "n_out": n_out,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
