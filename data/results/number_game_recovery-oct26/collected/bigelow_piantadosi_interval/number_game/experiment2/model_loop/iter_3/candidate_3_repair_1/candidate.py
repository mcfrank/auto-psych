"""Inferred stray-example rate on top of rule_size_prior_anchored_evidence.

Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian
concept learning (42 rules with a size-dependent prior and all intervals of
1..100, evidence tempered separately for rules and ranges, the first example's
evidence weighted by an inferred multiplier, a decision gain and shifted
criterion), but the rate at which they believe the program emits a stray number
outside its concept is inferred rather than fixed at 10%. The one change: each
example's likelihood under a concept of size |h| is (1 - eps) [x in h] / |h| +
eps / 100 with eps an inferred parameter; eps = 0.1 recovers the incumbent.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE = np.log(RULE_SIZE)
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()  # centred, for identifiability
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
ANCHOR = 0  # position of the anchoring example (first shown)

# Intervals of the same width containing the same number of examples (and the
# anchor or not) share a prior and a likelihood, so prepare_observed collapses
# each set's intervals into such groups; the posterior is exact. Likelihoods are
# built in the model from member counts, since the stray rate is inferred.

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    rule_first_in = pm.Data("rule_first_in", np.zeros((1, N_RULES)))  # first example inside
    set_n = pm.Data("set_n", np.ones(1))  # (sets,) number of examples
    group_width = pm.Data("group_width", np.ones((1, N_GROUPS)))  # (sets, groups)
    group_n_in = pm.Data("group_n_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_first_in = pm.Data("group_first_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    extra = anchor_weight - 1.0
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    # The single change: the believed stray-example rate is inferred (mean 0.1).
    eps = pm.Beta("eps", 2.0, 18.0)

    lp_out = pt.log(eps / 100.0)
    rule_lp_in = pt.log((1 - eps) / RULE_SIZE + eps / 100.0)  # (rules,)
    rule_ev = (
        rule_n_in * rule_lp_in[None, :] + (set_n[:, None] - rule_n_in) * lp_out
        + extra * (rule_first_in * rule_lp_in[None, :] + (1 - rule_first_in) * lp_out)
    )
    group_lp_in = pt.log((1 - eps) / group_width + eps / 100.0)
    group_ev = (
        group_n_in * group_lp_in + (set_n[:, None] - group_n_in) * lp_out
        + extra * (group_first_in * group_lp_in + (1 - group_first_in) * lp_out)
    )
    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    width = np.ones((n_sets, N_GROUPS))
    n_in = np.zeros((n_sets, N_GROUPS))
    first_in = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        anchored = INTERVAL_MEMBERSHIP[:, ex[ANCHOR]].astype("int64")
        group = ((INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside) * 2 + anchored
        for j, g in enumerate(np.unique(group)):  # compact: occupied groups only
            members = group == g
            k = np.argmax(members)
            log_count[i, j] = np.log(members.sum())
            width[i, j] = INTERVAL_WIDTH[k]
            n_in[i, j] = inside[k]
            first_in[i, j] = anchored[k]
            share[i, j] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    used = int((log_count > -700.0).sum(axis=1).max())
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_first_in": np.stack([RULE_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "group_width": width[:, :used],
        "group_n_in": n_in[:, :used],
        "group_first_in": first_in[:, :used],
        "group_log_count": log_count[:, :used],
        "group_in_concept": share[set_idx, :, tgt][:, :used],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
