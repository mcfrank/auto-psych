"""Refinement of log_number_line_interval_prior: people picture ranges on a
compressed mental number line, so a priori a range [m, n] among small numbers
is more plausible than an equally wide one among large numbers -- its prior is
multiplied by (m * n) ** (-compression), `compression` inferred (0 recovers a
magnitude-free prior; positive values make "yes" extend further below the
examples than above them). The one change is the learner underneath, now the
current best one (extended_multiples_concepts): 45 independently weighted rules
plus a multiples-of-13..50 family taking an inferred share `ext_share`, an
Erlang(2, sigma) prior on range size, an inferred share `noise` of random
examples, and examples drawn from the concept without replacement. "Yes" is the
posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Every interval [m, n], 1 <= m <= n <= 100.
INTERVAL_BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
N_INT = len(INTERVAL_BOUNDS)
INT_LO = np.array([m for m, _ in INTERVAL_BOUNDS])
INT_HI = np.array([n for _, n in INTERVAL_BOUNDS])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1
INT_SIZE = (INT_SIZE_IDX + 1).astype("float64")
SIZES_F = np.arange(1, 101, dtype="float64")
# Log magnitude of the endpoints, centred over intervals.
INT_LOG_MAG = np.log(INT_LO) + np.log(INT_HI)
INT_LOG_MAG = INT_LOG_MAG - INT_LOG_MAG.mean()
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
ALL_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    interval_cell = pm.Data("interval_cell", np.zeros((1, N_INT), dtype="int64"))  # k * N_INT + interval
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])

    # Interval prior: Erlang(2, sigma) on size, times (m * n) ** (-compression).
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    compression = pm.Normal("compression", 0.0, 1.0)
    log_int_prior = np.log(INT_SIZE) - INT_SIZE / sigma - compression * INT_LOG_MAG
    log_int_prior = log_int_prior - pt.logsumexp(log_int_prior)

    # Without replacement: the j-th in-concept example is one of size - j unseen
    # members. Likelihoods are worked relative to n_examples * log(noise / 100).
    log_out = pt.log(noise / 100.0)
    rule_log_lik = pt.zeros_like(rule_n_in)
    int_cum = [pt.zeros(100)]  # log-likelihood of the first k in-interval examples, by size
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0) - log_out
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
        int_cum.append(int_cum[-1] + pt.log((1 - noise) / np.maximum(SIZES_F - j, 1.0) + noise / 100.0) - log_out)
    # (k, intervals): prior times likelihood for k examples inside the interval.
    int_weight = pt.exp(pt.stack(int_cum)[:, INT_SIZE_IDX] + log_int_prior[None, :]) * (1 - lambda_0)
    int_unnorm = int_weight.reshape((-1,))[interval_cell]  # (sets, intervals)
    rule_unnorm = lambda_0 * all_weights[None, :] * pt.exp(rule_log_lik)
    unnorm = pt.concatenate([rule_unnorm, int_unnorm], axis=1)
    in_concept = pt.dot(unnorm, ALL_MEMBERSHIP) / unnorm.sum(axis=1, keepdims=True)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "interval_cell": np.stack([
            np.minimum(INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1), MAX_EX) * N_INT + np.arange(N_INT)
            for s in unique
        ]).astype("int64"),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
