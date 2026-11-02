"""Refinement of log_scale_interval_width_prior: people judge a range's width
on a logarithmic mental number line, so the prior over intervals [lo, hi] is
an Erlang(2, tau) density on L = log((hi + 0.5) / (lo - 0.5)) (tau inferred):
a typical range is narrow among small numbers and wide among large ones. The
one change is the learner underneath, now the current best one: 45
independently weighted rules (lambda_0 for the rules as a group), an inferred
share `noise` of random examples, and examples drawn from the concept without
replacement, so the j-th example inside a concept has likelihood
(1 - noise) / (size - j) + noise / 100. "Yes" is the posterior probability
that the concept contains the target.
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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

INTERVAL_BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
N_INT = len(INTERVAL_BOUNDS)
INT_LO = np.array([m for m, _ in INTERVAL_BOUNDS])
INT_HI = np.array([n for _, n in INTERVAL_BOUNDS])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1
SIZES_F = np.arange(1, 101, dtype="float64")
LOG_WIDTH = np.log((INT_HI + 0.5) / (INT_LO - 0.5))
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
ALL_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    interval_cell = pm.Data("interval_cell", np.zeros((1, N_INT), dtype="int64"))  # k * N_INT + interval
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Preferred log-width of a range: Erlang(2, tau) over log(hi / lo).
    tau = pm.LogNormal("tau", mu=np.log(0.5), sigma=1.0)
    log_erlang = np.log(LOG_WIDTH) - LOG_WIDTH / tau
    log_per_interval = log_erlang - pt.logsumexp(log_erlang)

    # Likelihoods are worked in linear space (they never fall below noise^4 / 10^8):
    # an interval's likelihood depends only on its size and the examples inside it.
    log_out = pt.log(noise / 100.0)
    rule_log_lik = (n_examples[:, None] - rule_n_in) * log_out
    int_cum = [pt.zeros(100)]  # log-likelihood of the first k in-interval examples, by size
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
        int_cum.append(int_cum[-1] + pt.log((1 - noise) / np.maximum(SIZES_F - j, 1.0) + noise / 100.0) - log_out)
    # (k, intervals): prior times likelihood, without the n_examples * log_out factor.
    int_weight = pt.exp(pt.stack(int_cum)[:, INT_SIZE_IDX] + log_per_interval[None, :]) * (1 - lambda_0)
    int_unnorm = int_weight.reshape((-1,))[interval_cell]  # (sets, intervals)
    rule_unnorm = lambda_0 * rule_weights[None, :] * pt.exp(rule_log_lik - n_examples[:, None] * log_out)
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
