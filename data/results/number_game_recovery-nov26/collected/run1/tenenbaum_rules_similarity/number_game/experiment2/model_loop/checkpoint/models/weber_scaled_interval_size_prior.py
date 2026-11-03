"""Bayesian concept learning with a family-weighted prior (exception rules in
their own family) and a Weber-scaled Erlang prior over interval size: a
refinement of `erlang_interval_size_family_prior`.

The learner considers 45 rules and the 5,050 intervals of 1..100, samples
examples from the concept (size principle, 10% uniformly from 1..100), and
answers "yes" with the posterior probability that the concept contains the
target. Each rule family gets an inferred prior mass shared by its members.
The one change: an interval's size is judged relative to its magnitude. An
interval of width s centred at c has effective size u = s / (c / 50)^gamma, and
its prior is proportional to u * exp(-u / sigma) (Erlang), with learned
characteristic size sigma and Weber exponent gamma (gamma = 0 recovers the
incumbent's absolute-width prior). So ranges among large numbers may be wider
a priori than ranges among small numbers.
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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
# Intervals live on a 100 x 100 (lower bound, upper bound) grid; cells with
# lower > upper are not intervals and get a large negative log weight.
LO, HI = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = HI >= LO
GRID_MASK = np.where(VALID, 0.0, -1e4)
INTERVAL_WIDTH = np.where(VALID, HI - LO + 1, 1).astype("float64")
LOG_REL_CENTRE = np.log((LO + HI) / 100.0)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Interval likelihoods P(D | lower, upper), scaled by each set's maximum
    # (grid_log_scale) and zero off the lower <= upper triangle.
    grid_lik = pm.Data("grid_lik", np.zeros((1, 100, 100)))  # (sets, lower, upper)
    grid_log_scale = pm.Data("grid_log_scale", np.zeros(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families (incl. exception rules) and of intervals.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    log_w = pt.log(family_weight)
    # Characteristic (magnitude-relative) interval size and Weber exponent.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    weber_exponent = pm.Beta("weber_exponent", alpha=1.5, beta=1.5)

    log_u = np.log(INTERVAL_WIDTH) - weber_exponent * LOG_REL_CENTRE
    erlang = log_u - pt.exp(log_u) / interval_scale + GRID_MASK
    log_per_interval = erlang - pt.logsumexp(erlang)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    interval_prior = pt.exp(log_per_interval)
    weighted = grid_lik * interval_prior[None, :, :]  # unnormalised interval posterior
    log_interval_mass = log_w[6] + grid_log_scale + pt.log(weighted.sum(axis=(1, 2)))
    log_norm = pt.logaddexp(pt.logsumexp(log_rules, axis=1), log_interval_mass)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    # P(target t in interval) = posterior mass of intervals with lower <= t <= upper.
    below = pt.cumsum(weighted, axis=1)  # (sets, t, upper): mass with lower <= t
    covered = (below * VALID).sum(axis=2)  # and upper >= t
    in_concept = rule_post @ pt.as_tensor(MEMBERSHIP) + covered * pt.exp(
        log_w[6] + grid_log_scale - log_norm
    )[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    grid_ll = np.zeros((len(unique), 100, 100))
    for i, s in enumerate(unique):
        x = np.array(s)[:, None, None]
        inside = (LO[None] <= x) & (x <= HI[None])
        grid_ll[i] = np.log(ALPHA * inside / INTERVAL_WIDTH[None] + (1 - ALPHA) / 100).sum(axis=0)
    scale = grid_ll.max(axis=(1, 2))
    return {
        "rule_log_lik": rule_ll,
        "grid_lik": np.exp(grid_ll - scale[:, None, None]) * VALID,
        "grid_log_scale": scale,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
