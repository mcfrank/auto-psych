"""Refinement of `bigelow_piantadosi_independent`: Bayesian concept learning
over 42 rules (each with an inferred prior weight) and every interval of 1-100,
with the size principle, where the share of examples people believe are random
draws from 1-100 (epsilon) is inferred rather than fixed at 10%. A larger
noise share lets a stray example be explained away, so rich example sets
yield tighter concepts around their bulk and less extreme judgments.

Because epsilon is a parameter, the likelihood of the example set under a
concept depends only on the concept's size and how many examples it contains.
prepare_observed groups the 5,050 intervals per set by (size, examples inside)
so the interval block costs one (groups x 100) product per set per gradient.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4  # at most four examples per set
N_K = MAX_N + 1
N_GROUPS = 100 * N_K  # (interval size 1..100) x (examples inside 0..4)


def concept_sets():
    """42 rules, then 5,050 intervals (as in the incumbent)."""
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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
GROUP_SIZE = np.repeat(np.arange(1, 101), N_K).astype("float64")  # size of group g
GROUP_K = np.tile(np.arange(N_K), 100).astype("float64")  # examples inside group g

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # log #intervals per group
    group_contains = pm.Data("group_contains", np.zeros((1, N_GROUPS, 100)))  # #intervals in group with target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    epsilon = pm.Beta("epsilon", 2, 4)  # believed share of random examples

    log_noise = pt.log(epsilon) - np.log(100.0)

    def log_lik(k, size, n):
        return k * pt.log((1 - epsilon) / size + epsilon / 100.0) + (n - k) * log_noise

    rule_ll = log_lik(rule_k, RULE_SIZE[None, :], n_examples[:, None])
    group_ll = log_lik(GROUP_K[None, :], GROUP_SIZE[None, :], n_examples[:, None]) + group_log_count
    interval_log_evidence = pt.logsumexp(group_ll, axis=1)  # (sets,)
    group_post = pt.exp(group_ll - interval_log_evidence[:, None])  # (sets, groups)
    interval_in_concept = pt.batched_dot(group_post, group_contains)  # (sets, 100)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_ll
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_k = np.zeros((len(unique), N_RULES))
    log_count = np.zeros((len(unique), N_GROUPS))
    contains = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        idx = np.array(s) - 1
        rule_k[i] = RULE_MEMBERSHIP[:, idx].sum(axis=1)
        k = INTERVAL_MEMBERSHIP[:, idx].sum(axis=1).astype("int64")
        group = (INTERVAL_SIZE - 1) * N_K + k
        onehot = np.zeros((N_GROUPS, N_INTERVALS))
        onehot[group, np.arange(N_INTERVALS)] = 1.0
        count = onehot.sum(axis=1)
        log_count[i] = np.where(count > 0, np.log(np.maximum(count, 1.0)), -1e4)
        contains[i] = onehot @ INTERVAL_MEMBERSHIP
    return {
        "rule_k": rule_k,
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        "group_log_count": log_count,
        "group_contains": contains,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
