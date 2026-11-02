"""Refinement of `set_size_amplified_size_principle`, adding one component from
`digit_sum_concepts_threshold_erlang`: digit-sum concepts as their own family.

People do the same Bayesian concept learning over 45 rules, bounded ranges of
1..100 (Erlang prior s * exp(-s / sigma) over width) and one-sided thresholds
([1, n], [m, 100]) in their own family, with the size principle (10% of
examples drawn at random from 1..100), a prior mass per concept family (the
exception rules in their own family) and each example's log-likelihood
weighted by k ** beta for a set of k examples. The one change: the hypothesis
space also holds "numbers whose digits add up to s" for s = 1..18, a new family
with its own inferred prior mass shared equally among the 18 sums, so examples
sharing a digit sum make other numbers with that sum more acceptable, and this
reading tightens with more examples as fast as any other tight concept does.
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
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
    if name.startswith("digit sum"):
        return 6
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
MEMBERSHIP = np.vstack([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])
CONCEPT_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
WIDTHS = np.arange(1, 99)  # widths a bounded range can have
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (INTERVAL_WIDTH == w)) for w in WIDTHS], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())
WIDTH_INDEX = np.where(THRESHOLD, 0, INTERVAL_WIDTH - 1)  # thresholds get a dummy index

with pm.Model() as model:
    # Per distinct set: summed log P(example | concept) over its examples, and log k.
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, len(MEMBERSHIP))))  # (sets, concepts)
    log_n_examples = pm.Data("log_n_examples", np.zeros((1, 1)), shape=(None, 1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families (digit sums the new one), bounded ranges and thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(9))
    log_w = pt.log(family_weight)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # How each example's weight scales with the number of examples shown.
    set_size_exponent = pm.Normal("set_size_exponent", mu=0.0, sigma=0.3)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    log_interval_prior = pt.switch(
        THRESHOLD, log_w[8] - np.log(N_THRESHOLD), log_w[7] + log_per_range[WIDTH_INDEX]
    )
    log_rule_prior = (log_w[:7] - np.log(FAMILY_SIZE))[RULE_FAMILY]
    log_prior = pt.concatenate([log_rule_prior, log_interval_prior])

    evidence_weight = pt.exp(set_size_exponent * log_n_examples)  # (sets, 1)
    log_joint = log_prior[None, :] + evidence_weight * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    return {
        "set_log_lik": np.stack([CONCEPT_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "log_n_examples": np.log([[len(s)] for s in unique]).astype("float64"),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
