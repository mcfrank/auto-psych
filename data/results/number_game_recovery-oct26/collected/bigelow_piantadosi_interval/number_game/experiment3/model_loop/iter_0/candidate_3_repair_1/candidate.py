"""Conservative, order-dependent belief updating with inflated concept sizes.

Refinement of inflated_size_familiar_rule_prior: the same Bayesian concept
learning (42 weighted rules with a size-power rule prior, all intervals of
1..100, 10% noisy examples, evidence tempered separately for rules and ranges,
first-example evidence multiplier, perceived size = true size + inflation,
decision gain and shifted criterion). The one change, taken from
conservative_partial_belief_updating: examples are read one at a time, and each
example after the first moves belief only an inferred fraction (update_rate) of
the way from the previous belief toward its full Bayesian update (a linear pool
of old and updated belief), so early-favoured broad concepts keep belief and
order matters; update_rate = 1 recovers the incumbent.
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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
LOG_NOISE = np.log(EPSILON / 100)

# Intervals of the same width containing the same examples (by position) share a
# prior and a whole sequential likelihood trajectory, so prepare_observed
# collapses each set's intervals into such groups; the filtering is exact.

with pm.Model() as model:
    rule_step_in = pm.Data("rule_step_in", np.zeros((1, MAX_SET, N_RULES)))  # (sets, step, rules)
    group_step_in = pm.Data("group_step_in", np.zeros((1, MAX_SET, 2)))  # (sets, step, groups)
    group_width = pm.Data("group_width", np.ones((1, 2)))  # (sets, groups)
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_SET)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 2)))
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 2)))  # (cells, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    size_inflation = pm.HalfNormal("size_inflation", sigma=10.0)
    # The one change: fraction of the way each later example moves belief.
    update_rate = pm.Beta("update_rate", 4.0, 1.5)

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    rule_in = in_lik(pt.as_tensor(RULE_SIZE))[None, None, :]
    rule_ev = rule_step_in * rule_in + (1 - rule_step_in) * LOG_NOISE
    group_in = in_lik(group_width)[:, None, :]
    group_ev = group_step_in * group_in + (1 - group_step_in) * LOG_NOISE
    step_ev = pt.concatenate(
        [rule_evidence_weight * rule_ev, range_evidence_weight * group_ev], axis=2
    )  # (sets, step, concepts)

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    log_prior = pt.concatenate(
        [
            pt.broadcast_to((pt.log(p_mix) + log_rule_prior)[None, :], (group_log_count.shape[0], N_RULES)),
            pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count,
        ],
        axis=1,
    )

    log_keep = pt.log1p(-update_rate)
    log_move = pt.log(update_rate)
    log_post = log_prior + anchor_weight * step_ev[:, 0, :]
    log_post = log_post - pt.logsumexp(log_post, axis=1, keepdims=True)
    for k in range(1, MAX_SET):
        m = step_mask[:, k][:, None]
        bayes = log_post + step_ev[:, k, :]
        bayes = bayes - pt.logsumexp(bayes, axis=1, keepdims=True)
        pooled = pt.logaddexp(log_keep + log_post, log_move + bayes)
        log_post = m * pooled + (1.0 - m) * log_post
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
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    per_set = []
    for s in unique:
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].astype("int64")
        pattern = (inside * (2 ** np.arange(len(ex)))[None, :]).sum(axis=1)
        group = (INTERVAL_WIDTH - 1) * (2 ** MAX_SET) + pattern
        per_set.append((ex, np.unique(group, return_inverse=True)))
    n_groups = max(len(u) for _, (u, _) in per_set)
    log_count = np.full((n_sets, n_groups), -700.0)
    width = np.ones((n_sets, n_groups))
    step_in = np.zeros((n_sets, MAX_SET, n_groups))
    share = np.zeros((n_sets, n_groups, 100))
    rule_in = np.zeros((n_sets, MAX_SET, N_RULES))
    mask = np.zeros((n_sets, MAX_SET))
    for i, (ex, (u, inv)) in enumerate(per_set):
        inv = inv.reshape(-1)
        counts = np.bincount(inv, minlength=len(u))
        log_count[i, : len(u)] = np.log(counts)
        first = np.zeros(len(u), dtype="int64")
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        width[i, : len(u)] = INTERVAL_WIDTH[first]
        for k, e in enumerate(ex):
            step_in[i, k, : len(u)] = INTERVAL_MEMBERSHIP[first, e]
            rule_in[i, k] = RULE_MEMBERSHIP[:, e]
            mask[i, k] = 1.0
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
        "step_mask": mask,
        "group_log_count": log_count,
        "group_in_concept": share[c_set, :, c_tgt],
        "cell_set": c_set.astype("int64"),
        "cell_flat": cells.astype("int64"),
        "row_cell": row_cell.astype("int64").reshape(-1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
