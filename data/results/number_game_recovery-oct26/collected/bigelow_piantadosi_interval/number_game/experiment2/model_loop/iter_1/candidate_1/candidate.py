"""Response perseveration (choice stickiness) on the simplest concept learner.

People do the same Bayesian concept learning as criterion_shifted_tempered_decision
(42 weighted rules and all 5,050 intervals of 1..100, size principle with 10%
noisy examples and evidence tempered by an inferred exponent) and map their
belief to a yes/no answer through an inferred gain on its log-odds and a
response bias, but the decision is sticky: the answer they gave on the
previous trial of the session shifts the criterion toward repeating it, by an
inferred perseveration strength. Concept inference is untouched; stickiness 0
recovers criterion_shifted_tempered_decision.
p_yes = sigmoid(gain * logit(P_w(target in concept | D)) + bias
                + stickiness * has_prev * (2 * previous_answer - 1)).
On a trial with no previous answer in the session (or a bare stimulus with no
history) the stickiness term is zero.
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
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so prepare_observed collapses each set's intervals
# into <= 500 groups (width x examples inside); the posterior is exact.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    # +1 if the previous answer in the session was yes, -1 if no, 0 if none.
    prev_answer = pm.Data("prev_answer", np.zeros(1, dtype="float64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # The single change: the examples' evidence is raised to this power.
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    # The single change: a response bias shifting the criterion on the log-odds.
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The single change: perseveration toward repeating the previous answer.
    stickiness = pm.Normal("stickiness", mu=0.0, sigma=1.0)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + evidence_weight * rule_log_lik
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + evidence_weight * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    # Decision gain on the log-odds of the belief, plus a criterion shift (bias).
    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias + stickiness * prev_answer), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    log_lik = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1)
        group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside
        for g in np.unique(group):
            members = group == g
            log_count[i, g] = np.log(members.sum())
            log_lik[i, g] = ll[members][0]
            share[i, g] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    # Previous answer within the same session (consecutive trial_index);
    # rows without a trial_index or rating (bare stimuli) have no history.
    n = len(rows)
    prev_answer = np.zeros(n)
    for i in range(1, n):
        cur, last = rows[i], rows[i - 1]
        if "trial_index" in cur and "trial_index" in last and "rating" in last:
            if int(float(cur["trial_index"])) == int(float(last["trial_index"])) + 1:
                prev_answer[i] = 2.0 * int(float(last["rating"])) - 1.0
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_count": log_count,
        "group_log_lik": log_lik,
        "group_in_concept": share[set_idx, :, tgt],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "prev_answer": prev_answer,
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
