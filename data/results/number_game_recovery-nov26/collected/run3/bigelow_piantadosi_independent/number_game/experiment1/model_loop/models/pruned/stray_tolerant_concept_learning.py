"""Refinement of `bigelow_piantadosi_mixture`: Bayesian concept learning over
the same 45 rules and every interval of 1..100 (rule chosen with probability
lambda_0, rules equally likely, each interval prior (1 - lambda_0) / 100^2),
with "yes" the posterior probability that the concept contains the target.

The one change: the rate at which learners believe an example is a stray
number drawn uniformly from 1..100 rather than from the concept is a free
parameter (1 - alpha) inferred from the data, not fixed at 10%. A learner who
tolerates more strays explains a scattered set like 92,82,27,42 as a tight
rule plus an outlier instead of a broad interval, and so generalises less to
numerical neighbours of the examples.

Each example x contributes log(alpha / |h| + (1 - alpha) / 100) if h contains
it and log((1 - alpha) / 100) otherwise, so intervals are pooled by (examples
inside, size), counted once per set in prepare_observed.
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


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZES = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZES = INTERVAL_MEMBERSHIP.sum(axis=1).astype(int)
MAX_K = 4  # at most four examples per set
# An interval's likelihood depends only on its size and how many examples it
# contains, so intervals are pooled into (count inside, size) groups:
# group index = k * 100 + size - 1, shared by all sets.
GROUP_SIZES = np.tile(np.arange(1, 101), MAX_K + 1).astype("float64")
GROUP_K = np.repeat(np.arange(MAX_K + 1), 100).astype("float64")

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside each rule
    group_count = pm.Data("group_count", np.zeros((1, 500)))  # (sets, groups): intervals per group
    group_target_count = pm.Data("group_target_count", np.zeros((100, 500)))  # (sets * 100, groups): of them, containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    alpha = pm.Beta("alpha", 6, 2)  # probability an example comes from the concept

    # Log-likelihood gain of an example inside h over one outside it (the
    # per-set term for examples outside every concept cancels in the posterior).
    log_out = pt.log((1 - alpha) / 100)
    rule_gain = pt.log(alpha / RULE_SIZES + (1 - alpha) / 100) - log_out
    group_log_w = GROUP_K * (pt.log(alpha / GROUP_SIZES + (1 - alpha) / 100) - log_out)
    w_max = pt.max(group_log_w)
    group_w = pt.exp(group_log_w - w_max)
    interval_evidence = group_count @ group_w  # (sets,), scaled by exp(-w_max)
    interval_in_concept = (group_target_count @ group_w).reshape((-1, 100)) / interval_evidence[:, None]

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_n_in * rule_gain[None, :]
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(interval_evidence) + w_max
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    group_count = np.zeros((len(unique), 500))
    group_target_count = np.zeros((len(unique), 100, 500))
    for i, s in enumerate(unique):
        k = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype(int)
        group = k * 100 + INTERVAL_SIZES - 1
        onehot = np.zeros((len(group), 500))
        onehot[np.arange(len(group)), group] = 1.0
        group_count[i] = onehot.sum(axis=0)
        group_target_count[i] = INTERVAL_MEMBERSHIP.T @ onehot
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_count": group_count,
        "group_target_count": group_target_count.reshape((-1, 500)),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
