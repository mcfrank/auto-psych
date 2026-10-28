"""Volatile-concept changepoint updating with final-digit union concepts.

Refinement of volatile_changepoint_inflated_size: people read the examples one
at a time, updating a Bayesian belief over 42 weighted rules (rule-size prior)
and all intervals of 1..100 while allowing, with an inferred switch
probability, that the program re-draws its concept between outputs (perceived
concept sizes inflated by a constant, tempered rule/range evidence, 10% noisy
examples, first-example evidence multiplier, decision gain and shifted
criterion as before). The one change: because people see numbers as written
digits, their concepts also include "ends in any of these digits" -- every
union of 2, 3 or 4 final digits (375 concepts) -- sharing an inferred slice
`union_share` of the rule prior (split equally across union arities, then
equally within one). After 2, 8 the union "ends in 2 or 8" is smaller than
"even" and less exotic than "powers of 2", so people endorse 62 or 82 more than
16; zero share recovers volatile_changepoint_inflated_size.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


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


def digit_unions():
    """Every union of 2, 3 or 4 final-digit classes, and its arity."""
    unions, arity = [], []
    for k in (2, 3, 4):
        for digits in itertools.combinations(range(10), k):
            unions.append([i for i in NUMBERS if i % 10 in digits])
            arity.append(k)
    return unions, np.array(arity)


RULES, INTERVALS = concept_sets()
UNIONS, UNION_ARITY = digit_unions()
UNION_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in UNIONS], dtype="float64")
# Log prior within the union slice: equal per arity, equal within an arity.
UNION_LOG_PRIOR = np.log(1.0 / 3.0) - np.log(np.array([np.sum(UNION_ARITY == k) for k in UNION_ARITY], dtype="float64"))
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE = np.log(RULE_SIZE)
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()

# Intervals of the same width containing the same examples (by position) share
# a prior and a whole sequential likelihood trajectory, so prepare_observed
# collapses each set's intervals into such groups; the filtering is exact.
# Every final-digit class has 10 members, so unions of the same arity holding
# the same examples are likewise collapsed into groups.

with pm.Model() as model:
    rule_step_in = pm.Data("rule_step_in", np.zeros((1, MAX_SET, N_RULES)))  # (sets, step, rules) membership
    group_step_in = pm.Data("group_step_in", np.zeros((1, MAX_SET, 2)))  # (sets, step, groups) membership
    group_width = pm.Data("group_width", np.ones((1, 2)))  # (sets, groups) interval width
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_SET)))  # 1 where the step is a real example
    union_step_in = pm.Data("union_step_in", np.zeros((1, MAX_SET, 2)))  # (sets, step, union groups)
    union_size = pm.Data("union_size", np.full((1, 2), 20.0))  # (sets, union groups)
    union_log_prior = pm.Data("union_log_prior", np.zeros((1, 2)))  # (sets, union groups), within the slice
    union_in_concept = pm.Data("union_in_concept", np.zeros((1, 2)))  # (cells, union groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 2)))  # (sets, groups)
    # Probabilities are computed once per distinct (set, target) cell, then
    # indexed out to the trials.
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 2)))  # (cells, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # trial -> cell
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # Believed probability that the program switches concept
    # (re-drawn from the prior) between consecutive outputs.
    switch_prob = pm.Beta("switch_prob", 1.5, 6.0)
    # Perceived concept size = true size + inflation.
    size_inflation = pm.HalfNormal("size_inflation", sigma=10.0)
    # The refinement: share of the rule prior on final-digit unions.
    union_share = pm.Beta("union_share", 1.0, 9.0)

    log_noise = np.log(EPSILON / 100)

    def member_loglik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    rule_step_lik = rule_step_in * member_loglik(RULE_SIZE)[None, None, :] + (1 - rule_step_in) * log_noise
    union_step_lik = union_step_in * member_loglik(union_size)[:, None, :] + (1 - union_step_in) * log_noise
    group_step_lik = group_step_in * member_loglik(group_width)[:, None, :] + (1 - group_step_in) * log_noise

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = pt.log1p(-union_share) + log_rule_prior - pt.logsumexp(log_rule_prior)
    log_prior = pt.concatenate(
        [
            pt.broadcast_to((pt.log(p_mix) + log_rule_prior)[None, :], (group_log_count.shape[0], N_RULES)),
            pt.log(p_mix) + pt.log(union_share) + union_log_prior,
            pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count,
        ],
        axis=1,
    )  # (sets, concepts)
    step_ev = pt.concatenate(
        [rule_evidence_weight * rule_step_lik, rule_evidence_weight * union_step_lik, range_evidence_weight * group_step_lik], axis=2
    )  # (sets, step, concepts)

    log_stay = pt.log1p(-switch_prob)
    log_switch = pt.log(switch_prob)
    # First example: prior times its (anchor-weighted) evidence.
    log_post = log_prior + anchor_weight * step_ev[:, 0, :]
    log_post = log_post - pt.logsumexp(log_post, axis=1, keepdims=True)
    for k in range(1, MAX_SET):
        m = step_mask[:, k][:, None]
        relaxed = pt.logaddexp(log_stay + log_post, log_switch + log_prior)
        updated = relaxed + step_ev[:, k, :]
        updated = updated - pt.logsumexp(updated, axis=1, keepdims=True)
        log_post = m * updated + (1.0 - m) * log_post
    posterior = pt.exp(log_post)

    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[cell_flat]
    n_u = union_size.shape[1]
    union_part = (posterior[:, N_RULES : N_RULES + n_u][cell_set] * union_in_concept).sum(axis=1)
    interval_part = (posterior[:, N_RULES + n_u :][cell_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + union_part + interval_part, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    cell_p = pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    p_yes = pm.Deterministic("p_yes", cell_p[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    per_set = []
    for s in unique:
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].astype("int64")  # (intervals, n_ex)
        pattern = (inside * (2 ** np.arange(len(ex)))[None, :]).sum(axis=1)
        group = (INTERVAL_WIDTH - 1) * (2 ** MAX_SET) + pattern
        per_set.append((ex, np.unique(group, return_inverse=True)))
    n_groups = max(len(u) for _, (u, _) in per_set)
    log_count = np.full((n_sets, n_groups), -700.0)
    step_in = np.zeros((n_sets, MAX_SET, n_groups))
    width = np.ones((n_sets, n_groups))
    share = np.zeros((n_sets, n_groups, 100))
    rule_in = np.zeros((n_sets, MAX_SET, N_RULES))
    mask = np.zeros((n_sets, MAX_SET))
    for i, (ex, (u, inv)) in enumerate(per_set):
        counts = np.bincount(inv, minlength=len(u))
        log_count[i, : len(u)] = np.log(counts)
        first = np.zeros(len(u), dtype="int64")
        first[inv[::-1]] = np.arange(len(inv))[::-1]  # a representative interval per group
        width[i, : len(u)] = INTERVAL_WIDTH[first]
        for k, e in enumerate(ex):
            step_in[i, k, : len(u)] = INTERVAL_MEMBERSHIP[first, e]
            rule_in[i, k] = RULE_MEMBERSHIP[:, e]
            mask[i, k] = 1.0
        np.add.at(share[i], inv, INTERVAL_MEMBERSHIP)
        share[i, : len(u)] /= counts[:, None]
    u_groups = []
    for ex, _ in per_set:
        upat = (UNION_MEMBERSHIP[:, ex].astype("int64") * (2 ** np.arange(len(ex)))[None, :]).sum(axis=1)
        u_groups.append(np.unique(UNION_ARITY * (2 ** MAX_SET) + upat, return_inverse=True))
    n_ug = max(len(u) for u, _ in u_groups)
    u_step = np.zeros((n_sets, MAX_SET, n_ug))
    u_size = np.full((n_sets, n_ug), 20.0)
    u_lp = np.full((n_sets, n_ug), -700.0)
    u_share = np.zeros((n_sets, n_ug, 100))
    for i, ((ex, _), (u, inv)) in enumerate(zip(per_set, u_groups)):
        counts = np.bincount(inv, minlength=len(u))
        first = np.zeros(len(u), dtype="int64")
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        u_size[i, : len(u)] = 10.0 * UNION_ARITY[first]
        u_lp[i, : len(u)] = UNION_LOG_PRIOR[first] + np.log(counts)
        for k, e in enumerate(ex):
            u_step[i, k, : len(u)] = UNION_MEMBERSHIP[first, e]
        np.add.at(u_share[i], inv, UNION_MEMBERSHIP)
        u_share[i, : len(u)] /= counts[:, None]
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    flat = set_idx * 100 + tgt
    cells, row_cell = np.unique(flat, return_inverse=True)
    c_set, c_tgt = cells // 100, cells % 100
    return {
        "rule_step_in": rule_in,
        "group_step_in": step_in,
        "group_width": width,
        "step_mask": mask,
        "group_log_count": log_count,
        "union_step_in": u_step,
        "union_size": u_size,
        "union_log_prior": u_lp,
        "union_in_concept": u_share[c_set, :, c_tgt],
        "group_in_concept": share[c_set, :, c_tgt],
        "cell_set": c_set.astype("int64"),
        "cell_flat": cells.astype("int64"),
        "row_cell": row_cell.astype("int64").reshape(-1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
