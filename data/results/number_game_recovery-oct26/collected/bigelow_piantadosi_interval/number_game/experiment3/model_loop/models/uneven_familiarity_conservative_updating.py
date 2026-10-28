"""Uneven rule familiarity on conservative, recency-weighted belief updating.

Refinement of conservative_updating_recency_boost: people read the examples one
at a time; after the first example (Bayesian concept learning over 42 rules
with a size-dependent prior and all intervals of 1..100, concept sizes inflated
by an inferred constant, one shared evidence weight, 10% noisy examples, a
first-example evidence multiplier, a decision gain and shifted criterion), each
later example moves belief only an inferred fraction of the way toward its full
Bayesian update, and the last example's update uses its own recency multiplier.
The one change, taken from uneven_rule_familiarity_ends: each rule's prior
familiarity is its size trend times an idiosyncratic factor drawn from a wide
log-normal spread, instead of Dirichlet(2) weights pulled toward one another,
so a few rules can be strongly familiar and many barely entertained.
"""

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


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE = np.log(RULE_SIZE)
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()
SIZE_DIR = RULE_LOG_SIZE_C / np.linalg.norm(RULE_LOG_SIZE_C)  # unit size-trend direction
FAMILIARITY_SPREAD = 3.0

# Intervals of the same width containing the same examples (by position) share
# a prior and a whole sequential likelihood trajectory, so prepare_observed
# collapses each set's intervals into such groups; the filtering is exact.

with pm.Model() as model:
    rule_step_in = pm.Data("rule_step_in", np.zeros((1, MAX_SET, N_RULES)))  # (sets, step, rules) membership
    group_step_in = pm.Data("group_step_in", np.zeros((1, MAX_SET, 2)))  # (sets, step, groups) membership
    group_width = pm.Data("group_width", np.ones((1, 2)))  # (sets, groups) interval width
    # Sets are ordered by size (largest first); n_active[k] is how many have a
    # k-th example, so each update step runs only on those sets.
    n_active = pm.Data("n_active", np.zeros(MAX_SET, dtype="int64"))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 2)))  # (sets, groups)
    # Probabilities are computed once per distinct (set, target) cell, then
    # indexed out to the trials.
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 2)))  # (cells, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # trial -> cell
    # 1 at the last position of sets with two or more examples, else 0.
    last_step = pm.Data("last_step", np.zeros((1, MAX_SET)))  # (sets, step)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    # The single change: idiosyncratic rule familiarity on the log scale, with a
    # wide fixed spread (Dirichlet(2) log-weights have sd ~0.8), kept orthogonal
    # to the size trend so it cannot trade off with size_exponent.
    familiarity_z = pm.ZeroSumNormal("familiarity_z", sigma=1.0, shape=N_RULES)
    familiarity = FAMILIARITY_SPREAD * (familiarity_z - pt.dot(familiarity_z, SIZE_DIR) * SIZE_DIR)
    # One evidence weight for rules and ranges (their separate weights were
    # nearly equal and trade off against the size inflation).
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # Fraction of the way each later example moves belief from the previous
    # belief toward its full Bayesian update.
    update_rate = pm.Beta("update_rate", 4.0, 1.5)
    # The refinement: perceived concept size = true size + inflation.
    # A log-normal prior keeps the posterior off the zero boundary, where a
    # half-normal gives NUTS a funnel against the evidence weight.
    size_inflation = pm.LogNormal("size_inflation", mu=np.log(3.0), sigma=0.75)

    log_noise = np.log(EPSILON / 100)

    def member_loglik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    rule_step_lik = rule_step_in * member_loglik(RULE_SIZE)[None, None, :] + (1 - rule_step_in) * log_noise
    group_step_lik = group_step_in * member_loglik(group_width)[:, None, :] + (1 - group_step_in) * log_noise

    log_rule_prior = familiarity + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    log_prior = pt.concatenate(
        [
            pt.broadcast_to((pt.log(p_mix) + log_rule_prior)[None, :], (group_log_count.shape[0], N_RULES)),
            pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count,
        ],
        axis=1,
    )  # (sets, concepts)
    step_ev = evidence_weight * pt.concatenate([rule_step_lik, group_step_lik], axis=2)  # (sets, step, concepts)

    # Recency: the last example's evidence has its own multiplier.
    recency_weight = pm.LogNormal("recency_weight", mu=0.0, sigma=0.75)
    step_ev = step_ev * (1.0 + (recency_weight - 1.0) * last_step)[:, :, None]

    log_keep = pt.log1p(-update_rate)
    log_move = pt.log(update_rate)
    # First example: prior times its (anchor-weighted) evidence.
    log_post = log_prior + anchor_weight * step_ev[:, 0, :]
    log_post = log_post - pt.logsumexp(log_post, axis=1, keepdims=True)
    for k in range(1, MAX_SET):
        n = n_active[k]
        old = log_post[:n]
        bayes = old + step_ev[:n, k, :]
        bayes = bayes - pt.logsumexp(bayes, axis=1, keepdims=True)
        log_post = pt.set_subtensor(log_post[:n], pt.logaddexp(log_keep + old, log_move + bayes))
    posterior = pt.exp(log_post)

    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[cell_flat]
    interval_part = (posterior[:, N_RULES:][cell_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    cell_p = pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    p_yes = pm.Deterministic("p_yes", cell_p[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = sorted(dict.fromkeys(sets), key=len, reverse=True)
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
    last_step = np.zeros((n_sets, MAX_SET))
    for i, (ex, (u, inv)) in enumerate(per_set):
        counts = np.bincount(inv, minlength=len(u))
        log_count[i, : len(u)] = np.log(counts)
        first = np.zeros(len(u), dtype="int64")
        first[inv[::-1]] = np.arange(len(inv))[::-1]  # a representative interval per group
        width[i, : len(u)] = INTERVAL_WIDTH[first]
        if len(ex) >= 2:
            last_step[i, len(ex) - 1] = 1.0
        for k, e in enumerate(ex):
            step_in[i, k, : len(u)] = INTERVAL_MEMBERSHIP[first, e]
            rule_in[i, k] = RULE_MEMBERSHIP[:, e]
        np.add.at(share[i], inv, INTERVAL_MEMBERSHIP)
        share[i, : len(u)] /= counts[:, None]
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    flat = set_idx * 100 + tgt
    cells, row_cell = np.unique(flat, return_inverse=True)
    c_set, c_tgt = cells // 100, cells % 100
    return {
        "rule_step_in": rule_in,
        "group_step_in": step_in,
        "group_width": width,
        "last_step": last_step,
        "n_active": np.array([sum(len(u) > k for u in unique) for k in range(MAX_SET)], dtype="int64"),
        "group_log_count": log_count,
        "group_in_concept": share[c_set, :, c_tgt],
        "cell_set": c_set.astype("int64"),
        "cell_flat": cells.astype("int64"),
        "row_cell": row_cell.astype("int64").reshape(-1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
