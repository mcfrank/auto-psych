"""Refinement of bigelow_piantadosi_independent with an Erlang interval-size prior.

People do Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 / with 37, multiples of 3-9, 11,
12, ending in 0-9, containing digit 0-9), each with its own inferred prior
weight and a group share lambda_0, plus every interval of 1..100. The single
change from the incumbent: the intervals share (1 - lambda_0) not uniformly but
by an Erlang density on their size, p(|h|) ~ |h|/sigma^2 exp(-|h|/sigma), with
sigma inferred, so intervals of a characteristic width are favoured and
generalization far outside the example range is curbed. Examples are sampled
from the concept (size principle) with 10% uniform noise; "yes" is the
posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVAL_BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in INTERVAL_BOUNDS]
INTERVAL_SIZE_IDX = np.array([n - m for m, n in INTERVAL_BOUNDS])  # size - 1
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

SIZES = np.arange(1, 101, dtype="float64")
N_OF_SIZE = 101.0 - SIZES

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of one size share a prior, so per example set they are grouped by
# size: log total evidence per size, and per size the in-concept probability
# of each target (evidence-weighted share of intervals of that size containing
# it). The posterior is exact over all 5,095 hypotheses.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes)
    size_in_concept = pm.Data("size_in_concept", np.zeros((1, 100, 100)))  # (sets, sizes, targets)
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
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    log_erlang = pt.log(SIZES) - 2 * log_sigma - SIZES / sigma
    log_size_prior = log_erlang - pt.logsumexp(np.log(N_OF_SIZE) + log_erlang)  # per interval

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_sizes = pt.log1p(-lambda_0) + log_size_prior[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + (posterior[:, N_RULES:, None] * size_in_concept).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    ilog = set_log_lik[:, N_RULES:]
    size_ev = np.full((len(unique), 100), -np.inf)
    size_ic = np.zeros((len(unique), 100, 100))
    for k in range(100):
        sel = INTERVAL_SIZE_IDX == k
        ev = logsumexp(ilog[:, sel], axis=1)
        size_ev[:, k] = ev
        size_ic[:, k, :] = np.exp(ilog[:, sel] - ev[:, None]) @ INTERVAL_MEMBERSHIP[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "size_in_concept": size_ic,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
