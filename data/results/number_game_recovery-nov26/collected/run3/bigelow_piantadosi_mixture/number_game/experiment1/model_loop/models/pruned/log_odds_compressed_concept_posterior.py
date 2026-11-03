"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning whose
posterior is reported through a linear-in-log-odds probability distortion.

The learner considers the 45 rule concepts of Bigelow & Piantadosi (2016) and
the 5,050 intervals of 1..100, with a rule share lambda_0, one prior weight per
rule, and each interval prior (1 - lambda_0) / 100^2. Examples are generated
from the concept (size principle, 10% uniform noise). The single change: the
posterior probability q that the concept contains the target is distorted
linearly in log-odds, logit(p_yes) = gamma * logit(q), with one inferred slope
gamma (1 = faithful report, < 1 = compression toward even odds).
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

# An interval [lo, hi] of size m with k of the n examples inside has
# log-likelihood k log(0.9 / m + 0.001) + (n - k) log(0.001): one of 2,000
# values indexed by (n, m, k). The intervals' total evidence for a set
# and the part from intervals containing a target are linear in
# exp(value), with interval counts per value that prepare_observed
# computes once. This is exact over all 5,050 intervals.
MAX_N = 4
LO, HI = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = HI >= LO
SIZE = HI - LO + 1
_N, _M, _K = np.meshgrid(np.arange(1, MAX_N + 1), NUMBERS, np.arange(MAX_N + 1), indexing="ij")
INTERVAL_LOG_LIK = (_K * np.log(ALPHA / _M + (1 - ALPHA) / 100) + (_N - _K) * np.log((1 - ALPHA) / 100)).ravel()
N_VALUES = INTERVAL_LOG_LIK.size

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def interval_counts(examples):
    """Interval counts per likelihood value: all intervals, and those containing each target."""
    k = sum(((LO <= x) & (x <= HI)).astype(int) for x in examples)
    value = ((len(examples) - 1) * 100 + SIZE - 1) * (MAX_N + 1) + k
    total = np.bincount(value[VALID], minlength=N_VALUES).astype("float64")
    by_target = np.stack([
        np.bincount(value[VALID & (LO <= t) & (t <= HI)], minlength=N_VALUES) for t in NUMBERS
    ]).astype("float64")
    return total, by_target


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    interval_count = pm.Data("interval_count", np.zeros((1, N_VALUES)))  # (sets, values)
    cell_interval_count = pm.Data("cell_interval_count", np.zeros((1, N_VALUES)))  # (set-target cells, values)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))  # set index of each trial
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set-target cell of each trial
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)  # without "contains digit 0"
    zero_split = pm.Beta("zero_split", 2, 2)  # "ends in 0"'s share of {10, 20, ..., 100}
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Probability-distortion slope in log-odds (1 = faithful report of the posterior).
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=0.5)

    interval_lik = pt.exp(pt.as_tensor(INTERVAL_LOG_LIK))  # P(D | interval) per value
    interval_evidence = pt.dot(interval_count, interval_lik)  # (sets,)
    interval_in_concept = pt.dot(cell_interval_count, interval_lik)[row_cell] / interval_evidence[row_set]

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(interval_evidence)
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    rule_in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_set * 100 + row_target]
    in_concept = rule_in_concept + posterior[row_set, N_RULES] * interval_in_concept

    q = pt.clip(in_concept, 1e-9, 1 - 1e-9)
    distorted = pm.math.sigmoid(gamma * (pt.log(q) - pt.log1p(-q)))
    p_yes = pm.Deterministic("p_yes", pt.clip(distorted, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = [interval_counts(s) for s in unique]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_log_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_count": np.stack([c[0] for c in counts]),
        "cell_interval_count": np.stack([counts[index[s]][1][t - 1] for s, t in cells]),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "row_target": np.array([t - 1 for t in targets], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
