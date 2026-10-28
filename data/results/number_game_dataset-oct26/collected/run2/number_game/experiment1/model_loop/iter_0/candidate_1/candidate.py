"""Bayesian concept learning from a recency-biased memory of the examples.

People are Bayesian concept learners (rules and intervals, size principle) who
reason only from the examples they still hold in memory: the last-shown example
is always remembered, and an example shown `lag` positions before it is
retained with probability retain**lag (forgotten otherwise). The yes
probability is the posterior probability that the concept contains the target,
averaged over which examples were remembered. This is order-dependent: targets
near the last example but outside the whole set's concept get more "yes" than
an order-invariant learner predicts.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SUB = 8  # 2**3 memory states of the (at most three) earlier examples
MAX_LAG = 3


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]


with pm.Model() as model:
    # Per remembered subset (row of the subset table): parameter-free evidence.
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    # Per distinct set: its memory states, which earlier examples each keeps.
    sub_index = pm.Data("sub_index", np.zeros((1, MAX_SUB), dtype="int64"))
    sub_valid = pm.Data("sub_valid", np.zeros((1, MAX_SUB)))
    kept = pm.Data("kept", np.zeros((1, MAX_SUB, MAX_LAG)))
    present = pm.Data("present", np.zeros((1, MAX_SUB, MAX_LAG)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    retain = pm.Beta("retain", 2, 2)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] * interval_in_concept

    # Probability of each memory state: retain**lag per kept earlier example.
    lags = pt.as_tensor(np.arange(1, MAX_LAG + 1, dtype="float64"))
    log_keep = lags * pt.log(pt.clip(retain, 1e-6, 1 - 1e-6))
    log_forget = pt.log1p(-pt.exp(log_keep))
    log_state = (present * (kept * log_keep + (1 - kept) * log_forget)).sum(axis=2)
    state_w = sub_valid * pt.exp(log_state)  # (sets, MAX_SUB), sums to 1 per set
    set_in_concept = (state_w[:, :, None] * in_concept[sub_index]).sum(axis=1)  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(set_in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    subsets = []
    sub_index = np.zeros((len(unique), MAX_SUB), dtype="int64")
    sub_valid = np.zeros((len(unique), MAX_SUB))
    kept = np.zeros((len(unique), MAX_SUB, MAX_LAG))
    present = np.zeros((len(unique), MAX_SUB, MAX_LAG))
    for si, s in enumerate(unique):
        n = len(s)
        earlier = n - 1  # example i (0-based) has lag n - 1 - i
        for k, mask in enumerate(itertools.product([0, 1], repeat=earlier)):
            chosen = [s[i] for i in range(earlier) if mask[i]] + [s[-1]]
            sub_index[si, k] = len(subsets)
            sub_valid[si, k] = 1.0
            for i in range(earlier):
                lag = n - 1 - i
                kept[si, k, lag - 1] = mask[i]
                present[si, k, lag - 1] = 1.0
            subsets.append(chosen)
        for k in range(2 ** earlier, MAX_SUB):
            sub_index[si, k] = sub_index[si, 0]
    sub_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(c) - 1].sum(axis=1) for c in subsets])
    interval_log_lik = sub_log_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": sub_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "sub_index": sub_index,
        "sub_valid": sub_valid,
        "kept": kept,
        "present": present,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
