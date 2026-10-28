"""Response perseveration on top of rule_size_prior_anchored_evidence.

Refinement of rule_size_prior_anchored_evidence: people do the same Bayesian
concept learning (42 weighted rules with a size-dependent prior and all intervals
of 1..100, evidence tempered separately for rules and ranges, 10% noisy examples,
the first example's evidence weighted by an inferred multiplier, a decision gain
and shifted criterion), but their yes/no criterion is pulled toward the answer
they gave on the immediately preceding trial of the session. The one change: an
inferred stickiness shifts the criterion by +stickiness after a "yes" and
-stickiness after a "no" (0 on a session's first trial); zero recovers the
refined model.
"""

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
RULE_LOG_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()  # centred, for identifiability
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
ANCHOR = 0  # position of the anchoring example (first shown)

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so prepare_observed collapses each set's intervals
# into groups (width x examples inside x contains the first example), keeping
# only the occupied groups; the posterior is exact.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    rule_anchor_lik = pm.Data("rule_anchor_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_anchor_lik = pm.Data("group_anchor_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    prev_response = pm.Data("prev_response", np.zeros(1))  # +1 prev yes, -1 prev no, 0 none
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # The single change: separate evidence tempering for rules and for ranges.
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    # Response bias shifting the criterion on the log-odds (as in the incumbent).
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The single change: the first example's evidence counts anchor_weight times
    # as much as a later example's (1 = no anchoring).
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    extra = anchor_weight - 1.0
    # The single change: rule prior proportional to rule size ** size_exponent,
    # renormalised within the rules (0 = the incumbent's prior).
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    rule_ev = rule_log_lik + extra * rule_anchor_lik
    group_ev = group_log_lik + extra * group_anchor_lik
    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    # Decision gain on the log-odds of the belief, plus a criterion shift (bias).
    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    # The single change: the criterion carries over toward the previous answer.
    stickiness = pm.Normal("stickiness", mu=0.0, sigma=0.5)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias + stickiness * prev_response), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `trial_index` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    log_lik = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    anchor_lik = np.zeros((n_sets, N_GROUPS))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1)
        a_ll = EXAMPLE_LOG_LIK[N_RULES:, ex[ANCHOR]]
        anchored = INTERVAL_MEMBERSHIP[:, ex[ANCHOR]].astype("int64")
        group = ((INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside) * 2 + anchored
        for j, g in enumerate(np.unique(group)):  # compact: occupied groups only
            members = group == g
            log_count[i, j] = np.log(members.sum())
            log_lik[i, j] = ll[members][0]
            share[i, j] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
            anchor_lik[i, j] = a_ll[members][0]
    used = int((log_count > -700.0).sum(axis=1).max())
    log_count, log_lik, anchor_lik = log_count[:, :used], log_lik[:, :used], anchor_lik[:, :used]
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    ratings = np.array([int(r["rating"]) for r in rows], dtype="int64")
    trial = np.array([int(r["trial_index"]) for r in rows], dtype="int64")
    prev = np.zeros(len(rows))
    consecutive = trial[1:] == trial[:-1] + 1  # same session, next trial
    prev[1:] = np.where(consecutive, 2.0 * ratings[:-1] - 1.0, 0.0)
    return {
        "prev_response": prev,
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, s[ANCHOR] - 1] for s in unique]),
        "group_anchor_lik": anchor_lik,
        "group_log_count": log_count,
        "group_log_lik": log_lik,
        "group_in_concept": share[set_idx, :, tgt][:, :used],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": ratings,
    }
