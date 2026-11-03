"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
45 independently weighted rules and all intervals of 1..100 (size principle,
10% of examples drawn uniformly from 1..100), where the prior over intervals is
not uniform but follows an Erlang density on interval size,
p(size) proportional to size * exp(-size / sigma), with sigma (the preferred
interval width) inferred. The intervals as a group keep prior mass
1 - lambda_0, shared among them in proportion to that Erlang weight. "Yes" is
the posterior probability that the concept contains the target.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
INTERVAL_SIZE = np.array([len(c) for c in INTERVALS])
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
SIZES = np.arange(1, 101, dtype="float64")
N_PER_SIZE = 101.0 - SIZES  # intervals of each size in 1..100

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of one size share a prior, so per example set they are grouped by
# size: summed evidence per size, and per target the share of that evidence
# from intervals of that size containing it (computed in prepare_observed).

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
    # Preferred interval width: Erlang(2, sigma) over interval size.
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES) - SIZES / sigma
    # Log prior of one interval of each size; intervals sum to 1 - lambda_0.
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_sizes = pt.log1p(-lambda_0) + log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * size_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    interval_membership = MEMBERSHIP[N_RULES:]
    size_ev = np.zeros((len(unique), 100))
    size_in = np.zeros((len(unique), 100, 100))
    for k in range(100):
        sel = INTERVAL_SIZE == k + 1
        ll = interval_log_lik[:, sel]
        ev = logsumexp(ll, axis=1)
        size_ev[:, k] = ev
        size_in[:, k, :] = np.exp(ll - ev[:, None]) @ interval_membership[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "size_in_concept": size_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
