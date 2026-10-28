"""Stochastic forgetting of examples before Bayesian concept learning.

People do the same Bayesian concept learning as criterion_shifted_tempered_decision
(42 weighted rules and all intervals of 1..100, size principle with 10% noisy
examples, evidence tempered by an inferred exponent, and an inferred decision gain
and criterion shift on the belief's log-odds), but each example is retained in
memory only with an inferred probability `recall`, independently, and the concept
is inferred from the retained subset. Belief that the target is in the concept is
the average of the subset posteriors' predictions, weighted by
recall^|S| (1 - recall)^(n - |S|); recall = 1 recovers the incumbent.
p_yes = sigmoid(gain * logit(sum_S w_S P_w(target in concept | S)) + bias).
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    """42 rules, then 5,050 intervals."""
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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)
N_SUB = 2 ** MAX_SET  # remembered subsets per set, padded

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so each remembered subset's intervals collapse into
# <= 500 groups (width x examples inside); the posterior is exact.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (subsets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (subsets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (subsets, groups)
    # Each (remembered subset, target) pair lists its occupied interval groups
    # (padded): their flat index into (subsets, groups) and the share of the
    # group's intervals that contain the target.
    pair_group = pm.Data("pair_group", np.zeros((1, 1), dtype="int64"))  # (pairs, max groups)
    pair_share = pm.Data("pair_share", np.zeros((1, 1)))  # (pairs, max groups)
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))  # subset index * 100 + target - 1
    slot_pair = pm.Data("slot_pair", np.zeros(1, dtype="int64"))  # (cells * N_SUB,)
    slot_kept = pm.Data("slot_kept", np.zeros(1))  # examples retained
    slot_lost = pm.Data("slot_lost", np.zeros(1))  # examples forgotten
    slot_valid = pm.Data("slot_valid", np.zeros(1))  # 0 for padding
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The single change: each example is retained with this probability.
    recall = pm.Beta("recall", 6.0, 1.5)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + evidence_weight * rule_log_lik
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + evidence_weight * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_pred = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)  # (subsets, 100)
    interval_post = posterior[:, N_RULES:].reshape((-1,))
    pair_p = rule_pred.reshape((-1,))[pair_cell] + (interval_post[pair_group] * pair_share).sum(axis=1)
    slot_p = pair_p[slot_pair]
    slot_w = slot_valid * pt.exp(slot_kept * pt.log(recall) + slot_lost * pt.log1p(-recall))
    cell_p = (slot_w * slot_p).reshape((-1, N_SUB)).sum(axis=1)
    concept_p = pt.clip(cell_p[row_cell], 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _subset_tables(subset):
    ex = np.array(subset, dtype="int64") - 1
    rule_ll = EXAMPLE_LOG_LIK[:N_RULES, ex].sum(axis=1) if len(ex) else np.zeros(N_RULES)
    inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64") if len(ex) else np.zeros(len(INTERVALS), "int64")
    ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1) if len(ex) else np.zeros(len(INTERVALS))
    group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside
    log_count = np.full(N_GROUPS, -700.0)  # empty groups get no mass
    log_lik = np.zeros(N_GROUPS)
    share = np.zeros((N_GROUPS, 100))
    for g in np.unique(group):
        members = group == g
        log_count[g] = np.log(members.sum())
        log_lik[g] = ll[members][0]
        share[g] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    return rule_ll, log_count, log_lik, share


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    tgts = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, tgts)))
    cell_index = {c: i for i, c in enumerate(cells)}
    subsets, sub_index, tables = [], {}, []
    slot_cell, slot_kept, slot_lost, slot_valid = [], [], [], []
    for s, t in cells:
        n = len(s)
        combos = [c for k in range(n + 1) for c in itertools.combinations(range(n), k)]
        for j in range(N_SUB):
            if j < len(combos):
                key = tuple(sorted(s[i] for i in combos[j]))
                valid, kept = 1.0, len(combos[j])
            else:
                key, valid, kept = (), 0.0, 0
            if key not in sub_index:
                sub_index[key] = len(subsets)
                subsets.append(key)
                tables.append(_subset_tables(key))
            k = sub_index[key]
            slot_cell.append(k * 100 + t - 1)
            slot_kept.append(float(kept))
            slot_lost.append(float(n - kept) if valid else 0.0)
            slot_valid.append(valid)
    slot_cell = np.array(slot_cell, dtype="int64")
    pairs = np.unique(slot_cell)
    nonzero = [np.nonzero(tables[c // 100][3][:, c % 100])[0] for c in pairs]
    width = max(len(g) for g in nonzero)
    pair_group = np.zeros((len(pairs), width), dtype="int64")
    pair_share = np.zeros((len(pairs), width))
    for i, (c, g) in enumerate(zip(pairs, nonzero)):
        pair_group[i, : len(g)] = (c // 100) * N_GROUPS + g
        pair_share[i, : len(g)] = tables[c // 100][3][g, c % 100]
    return {
        "rule_log_lik": np.stack([tb[0] for tb in tables]),
        "group_log_count": np.stack([tb[1] for tb in tables]),
        "group_log_lik": np.stack([tb[2] for tb in tables]),
        "pair_group": pair_group,
        "pair_share": pair_share,
        "pair_cell": pairs,
        "slot_pair": np.searchsorted(pairs, slot_cell).astype("int64"),
        "slot_kept": np.array(slot_kept),
        "slot_lost": np.array(slot_lost),
        "slot_valid": np.array(slot_valid),
        "row_cell": np.array([cell_index[c] for c in zip(sets, tgts)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
