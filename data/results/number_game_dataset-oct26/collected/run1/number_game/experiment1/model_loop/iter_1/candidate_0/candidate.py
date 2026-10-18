"""Win-stay, lose-shift concept learning.

People hold a single concept at a time. They sample a concept from the
posterior given the first example, keep it as long as each new example fits
it, and only when an example contradicts it do they shift to a concept
sampled from the posterior given all examples seen so far; they say "yes"
exactly when the target belongs to the concept they currently hold.
Consistent examples never narrow the held concept (no size-principle
sharpening after the first example), and judgments depend on example order.

The hypothesis space is the 42 rules (inferred prior weights, group share
p_mix) plus every interval of 1-100 (equal prior), with the size principle
and an inferred share epsilon of examples believed to be random draws. The
distribution over the held concept after reading the examples is computed
exactly: q_1 = post(x_1); q_k = q_{k-1} * [x_k in h] + (mass shifted) *
post(x_1..x_k).

Intervals with the same size and the same subset of the examples inside
them behave identically (same prior, likelihood and stay/shift history), so
prepare_observed groups them per set (at most a few hundred groups) and the
model tracks one probability per group.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1)
N_G = 480  # max interval groups per set (473 on this design), padded

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, MAX_N, N_RULES)))  # prefix examples inside rule
    rule_stay = pm.Data("rule_stay", np.ones((1, MAX_N, N_RULES)))  # example k fits rule
    group_k = pm.Data("group_k", np.zeros((1, MAX_N, N_G)))  # prefix examples inside group
    group_stay = pm.Data("group_stay", np.ones((1, MAX_N, N_G)))
    group_size = pm.Data("group_size", np.ones((1, N_G)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_G)))  # log #intervals (-1e4 pad)
    group_frac = pm.Data("group_frac", np.zeros((1, N_G, 100)))  # share of group containing t
    prefix_n = pm.Data("prefix_n", np.ones((1, MAX_N)))
    real_step = pm.Data("real_step", np.zeros((1, MAX_N)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    epsilon = pm.Beta("epsilon", 2, 4)

    log_out = pt.log(epsilon) - np.log(100.0)
    n = prefix_n[:, :, None]
    rule_ll = rule_k * pt.log((1 - epsilon) / RULE_SIZE + epsilon / 100.0)[None, None, :] + (n - rule_k) * log_out
    group_ll = group_k * pt.log((1 - epsilon) / group_size + epsilon / 100.0)[:, None, :] + (n - group_k) * log_out
    log_joint = pt.concatenate([
        rule_ll + (pt.log(p_mix) + pt.log(rule_weights))[None, None, :],
        group_ll + (group_log_count + pt.log1p(-p_mix) - np.log(N_INTERVALS))[:, None, :],
    ], axis=2)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=2, keepdims=True))  # (S, K, R+G)
    stay = pt.concatenate([rule_stay, group_stay], axis=2)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = q * stay[:, k, :]
        shifted = 1.0 - kept.sum(axis=1, keepdims=True)
        q_new = kept + shifted * post[:, k, :]
        r = real_step[:, k][:, None]
        q = r * q_new + (1 - r) * q

    in_concept = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + pt.batched_dot(q[:, N_RULES:], group_frac)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` (in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    out = {
        "rule_k": np.zeros((S, MAX_N, N_RULES)), "rule_stay": np.ones((S, MAX_N, N_RULES)),
        "group_k": np.zeros((S, MAX_N, N_G)), "group_stay": np.ones((S, MAX_N, N_G)),
        "group_size": np.ones((S, N_G)), "group_log_count": np.full((S, N_G), -1e4),
        "group_frac": np.zeros((S, N_G, 100)), "prefix_n": np.ones((S, MAX_N)),
        "real_step": np.zeros((S, MAX_N)),
    }
    for i, s in enumerate(unique):
        idx = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, idx]  # (intervals, n)
        keys = {}
        for j in range(N_INTERVALS):
            keys.setdefault((INTERVAL_SIZE[j],) + tuple(inside[j]), []).append(j)
        assert len(keys) <= N_G
        for g, (key, members) in enumerate(keys.items()):
            pattern = np.array(key[1:])
            out["group_size"][i, g] = key[0]
            out["group_log_count"][i, g] = np.log(len(members))
            out["group_frac"][i, g] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
            for k in range(MAX_N):
                kk = min(k, len(s) - 1)
                out["group_k"][i, k, g] = pattern[: kk + 1].sum()
                if k < len(s):
                    out["group_stay"][i, k, g] = pattern[k]
        for k in range(MAX_N):
            kk = min(k, len(s) - 1)
            out["rule_k"][i, k] = RULE_MEMBERSHIP[:, idx[: kk + 1]].sum(axis=1)
            out["prefix_n"][i, k] = kk + 1
            if k < len(s):
                out["rule_stay"][i, k] = RULE_MEMBERSHIP[:, idx[k]]
                out["real_step"][i, k] = 1.0
    out["row_cell"] = np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
