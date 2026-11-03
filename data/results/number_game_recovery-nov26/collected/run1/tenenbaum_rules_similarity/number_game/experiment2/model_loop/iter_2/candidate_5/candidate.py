"""Refinement of `centre_weighted_range_sampling`: one-sided thresholds as their
own concept family (the component taken from `threshold_concepts_erlang_interval`).

People do the same Bayesian concept learning over 45 rules and the ranges of
1..100: a prior mass per concept family (exception rules in their own family),
the size principle (10% of examples drawn uniformly from 1..100), an Erlang
prior s * exp(-s / sigma) over the width of bounded ranges, and centre-weighted
sampling from a bounded range (density proportional to (u * (1 - u)) ** kappa
at relative position u). The one change: one-sided thresholds [1, n] and
[m, 100] are a separate kind of program with their own inferred prior mass,
shared equally among them, exempt from the width prior and sampled uniformly
(a threshold has no centre). "Yes" is the posterior probability that the
concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev
from scipy.special import logsumexp

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
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
THRESHOLD = (LO == 1) | (HI == 100)
N_THRESHOLD = float(THRESHOLD.sum())
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
N_BOUNDED_PER_WIDTH = np.array([np.sum(~THRESHOLD & (WIDTH == w)) for w in range(1, 101)], dtype="float64")
BOUNDED_WIDTHS = N_BOUNDED_PER_WIDTH > 0
CHEB_DEGREE = 6
# log(u * (1 - u)) at the k-th of W positions of a width-W range, u = (k - 0.5) / W.
_K = np.arange(1, 101)[None, :]
_U = np.clip((_K - 0.5) / WIDTHS[:, None], 1e-9, 1 - 1e-9)
SHAPE_TABLE = np.where(_K <= WIDTHS[:, None], np.log(_U * (1 - _U)), 0.0)
SHAPE_MASK = np.where(_K <= WIDTHS[:, None], 0.0, -1e4)


def example_log_lik_table(kappa):
    """Log P(example at position k of a width-W range), (W, k), for centre weight kappa."""
    log_density = kappa * SHAPE_TABLE - logsumexp(kappa * SHAPE_TABLE + SHAPE_MASK, axis=1, keepdims=True)
    return np.log(ALPHA * np.exp(log_density) + (1 - ALPHA) / 100)


# A bounded range's log likelihood is a smooth function of kappa alone, so each
# table entry is expanded in Chebyshev polynomials of 2 * kappa - 1 on [0, 1];
# the last row is "outside the range".
_NODES = np.cos(np.pi * (np.arange(30) + 0.5) / 30)
_TABLES = np.stack([example_log_lik_table((x + 1) / 2).reshape(-1) for x in _NODES])
CHEB_TABLE = np.concatenate(
    [
        chebyshev.chebfit(_NODES, _TABLES, CHEB_DEGREE).T,
        np.eye(1, CHEB_DEGREE + 1) * np.log((1 - ALPHA) / 100),
    ]
)  # (100 * 100 + 1, degree + 1)
OUTSIDE = 100 * 100

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set and range, Chebyshev coefficients (in 2 * kappa - 1) of the log
    # likelihood of the set's examples; thresholds have only a constant term.
    interval_cheb = pm.Data("interval_cheb", np.zeros((1, len(LO), CHEB_DEGREE + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families, bounded ranges and one-sided thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # How strongly a bounded range's outputs are expected to cluster at its centre.
    centre_weight = pm.Beta("centre_weight", alpha=1.5, beta=6.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang[BOUNDED_WIDTHS] + np.log(N_BOUNDED_PER_WIDTH[BOUNDED_WIDTHS]))
    log_prior_interval = pt.where(
        THRESHOLD, log_w[7] - np.log(N_THRESHOLD), log_w[6] + log_per_range[WIDTH - 1]
    )

    x = 2.0 * centre_weight - 1.0
    cheb = [pt.ones(()), x]
    for _ in range(CHEB_DEGREE - 1):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    interval_log_lik = pt.tensordot(interval_cheb, pt.stack(cheb), axes=[[2], [0]])  # (sets, intervals)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_intervals = log_prior_interval[None, :] + interval_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] @ pt.as_tensor(
        INTERVAL_MEMBERSHIP
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_cheb = np.zeros((len(unique), len(LO), CHEB_DEGREE + 1))
    for i, s in enumerate(unique):
        for x in s:
            hit = (LO <= x) & (x <= HI)
            interval_cheb[i] += CHEB_TABLE[np.where(hit, (WIDTH - 1) * 100 + (x - LO), OUTSIDE)]
        # Thresholds are sampled uniformly, whatever the centre weight.
        members = INTERVAL_MEMBERSHIP[THRESHOLD][:, np.array(s) - 1]
        uniform = np.log(ALPHA * members / WIDTH[THRESHOLD][:, None] + (1 - ALPHA) / 100).sum(axis=1)
        interval_cheb[i, THRESHOLD] = 0.0
        interval_cheb[i, THRESHOLD, 0] = uniform
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_cheb": interval_cheb,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
