"""Refinement of `threshold_concepts_erlang_interval`: a blunted size principle
for magnitude concepts.

People do the same Bayesian concept learning over 45 number rules, bounded
ranges [m, n] (Erlang prior s * exp(-s / sigma) over width) and one-sided
thresholds ([1, n], [m, 100], their own family), with a prior mass per concept
family and 10% of examples taken as random draws from 1..100. The one change:
the size principle is weaker for magnitude concepts than for rules. An example
inside a range of size s has probability s^-g * 100^(g-1) under it, with g
learned (g = 1 recovers the incumbent; g = 0 is weak sampling, the range judged
only by whether it covers the examples), while rules keep 1 / |rule|. So ranges
do not shrink onto clustered examples as fast as an ideal learner's would, and
"yes" extends further past the examples' extremes for multi-example sets.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9


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
# Bounded ranges [m, n] with 1 < m <= n < 100, and one-sided thresholds [1, n] or [m, 100].
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
INTERVAL_WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)  # widths a bounded range can have
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (INTERVAL_WIDTH == w)) for w in WIDTHS], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())
CONCEPT_MEMBERSHIP = np.vstack([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])
N_CONCEPTS = len(CONCEPT_MEMBERSHIP)
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
LOG_INTERVAL_SIZE = np.log(INTERVAL_WIDTH.astype("float64"))
BOUNDED_WIDTH_INDEX = np.where(THRESHOLD, 0, INTERVAL_WIDTH - 1)

with pm.Model() as model:
    # Per set and concept: how many examples the concept contains and how many it misses.
    n_inside = pm.Data("n_inside", np.zeros((1, N_CONCEPTS)))
    n_outside = pm.Data("n_outside", np.zeros((1, N_CONCEPTS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families, bounded ranges and one-sided thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Strength of the size principle for ranges and thresholds (1 = as for rules).
    range_size_strength = pm.Beta("range_size_strength", alpha=2.0, beta=2.0)

    # Probability of an example under each concept that contains it.
    log_density = pt.concatenate(
        [-LOG_RULE_SIZE, -range_size_strength * LOG_INTERVAL_SIZE + (range_size_strength - 1) * np.log(100.0)]
    )
    log_in = pt.log(ALPHA * pt.exp(log_density) + (1 - ALPHA) / 100)
    log_lik = n_inside * log_in[None, :] + n_outside * np.log((1 - ALPHA) / 100)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    interval_prior = pt.switch(
        THRESHOLD, log_w[7] - np.log(N_THRESHOLD), log_w[6] + log_per_range[BOUNDED_WIDTH_INDEX]
    )
    rule_prior = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY]
    log_joint = pt.concatenate([rule_prior, interval_prior])[None, :] + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(CONCEPT_MEMBERSHIP)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_inside = np.stack([CONCEPT_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_outside = np.array([len(s) for s in unique], dtype="float64")[:, None] - n_inside
    return {
        "n_inside": n_inside,
        "n_outside": n_outside,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
