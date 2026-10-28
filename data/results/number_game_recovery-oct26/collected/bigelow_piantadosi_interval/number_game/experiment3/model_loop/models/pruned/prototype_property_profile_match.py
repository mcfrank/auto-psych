"""Prototype property-profile matching.

People do not infer which concept generated the examples; they abstract a
prototype of the example set: for every familiar number property (42 arithmetic
rules and all ranges of 1..100) the prototype holds how strongly the set has
that property -- the fraction of examples with it, sharpened by an inferred
exponent -- times the property's salience, which falls as an inferred power of
how many numbers have it (rare properties are distinctive) and an accessibility
weight per rule. A target is endorsed in proportion to the share of the
prototype's total weight carried by properties the target also has, passed
through a decision gain and criterion. Salience is fixed, not compounded per
example as in the Bayesian size principle, and partially shared properties
still contribute.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


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
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
RULE_LOG_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)
LOG_FLOOR = -30.0  # log of the fraction for properties no example has (weight ~0)

with pm.Model() as model:
    rule_log_frac = pm.Data("rule_log_frac", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_frac = pm.Data("group_log_frac", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_width = pm.Data("group_log_width", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_share = pm.Data("group_share", np.zeros((1, N_GROUPS, 100)))  # (sets, groups, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_rule = pm.Beta("p_rule", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_salience = pm.HalfNormal("rule_salience", sigma=2.0)  # exponent on 1/size
    range_salience = pm.HalfNormal("range_salience", sigma=2.0)
    sharpness = pm.LogNormal("sharpness", mu=1.0, sigma=1.0)  # exponent on fraction shared
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    log_w_rules = (
        pt.log(p_rule) + pt.log(rule_weights)[None, :]
        - rule_salience * pt.as_tensor(RULE_LOG_SIZE)[None, :]
        + sharpness * rule_log_frac
    )
    log_w_groups = (
        pt.log1p(-p_rule) - np.log(len(INTERVALS)) + group_log_count
        - range_salience * group_log_width + sharpness * group_log_frac
    )
    log_w = pt.concatenate([log_w_rules, log_w_groups], axis=1)
    w = pt.exp(log_w - pt.logsumexp(log_w, axis=1, keepdims=True))  # (sets, props)
    rule_match = w[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)  # (sets, targets)
    range_match = pt.batched_dot(w[:, N_RULES:], group_share)  # (sets, targets)
    match = (rule_match + range_match).reshape((-1,))[row_cell]
    match = pt.clip(match, 1e-6, 1 - 1e-6)
    belief_logit = pt.log(match) - pt.log1p(-match)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _log_frac(k, n):
    return np.where(k > 0, np.log(np.maximum(k, 1) / n), LOG_FLOOR)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    g_frac = np.full((n_sets, N_GROUPS), LOG_FLOOR)
    g_width = np.zeros((n_sets, N_GROUPS))
    g_count = np.full((n_sets, N_GROUPS), -700.0)
    share = np.zeros((n_sets, N_GROUPS, 100))
    r_frac = np.zeros((n_sets, N_RULES))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n = len(s)
        r_frac[i] = _log_frac(RULE_MEMBERSHIP[:, ex].sum(axis=1), n)
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside
        for j, g in enumerate(np.unique(group)):
            members = group == g
            first = np.flatnonzero(members)[0]
            g_count[i, j] = np.log(members.sum())
            g_width[i, j] = np.log(INTERVAL_WIDTH[first])
            g_frac[i, j] = _log_frac(inside[first], n)
            share[i, j] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    used = int((g_count > -700.0).sum(axis=1).max())
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_frac": r_frac,
        "group_log_frac": g_frac[:, :used],
        "group_log_width": g_width[:, :used],
        "group_log_count": g_count[:, :used],
        "group_share": share[:, :used],
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
