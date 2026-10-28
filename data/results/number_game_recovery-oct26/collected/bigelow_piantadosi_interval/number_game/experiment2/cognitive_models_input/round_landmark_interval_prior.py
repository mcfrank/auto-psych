"""Round decade landmarks in the interval prior.

People do Bayesian concept learning over 42 weighted rules and all 5,050
intervals of 1..100 (size principle, 10% noisy examples, evidence tempered by
an inferred exponent), and map the posterior belief P(target in concept) to a
yes/no answer through an inferred gain and criterion shift on its log-odds, as
in criterion_shifted_tempered_decision. The single new mechanism: intervals
are not equally likely a priori. Intervals whose both boundaries sit on round
decade landmarks (lower bound 1 or ending in 0/1, upper bound 100 or ending in
9/0: e.g. 1-50, 20-29, 61-100, 80-89) get prior weight multiplied by an
inferred factor exp(round_boost) relative to intervals with arbitrary edges;
round_boost = 0 recovers the best model.
p_yes = sigmoid(gain * logit(P_w(target in concept | D)) + bias).
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
MAX_SET = 4
N_BASE_GROUPS = 100 * (MAX_SET + 1)
N_GROUPS = 2 * N_BASE_GROUPS  # second half: intervals with round boundaries


def _is_round(lo, hi):
    return (lo == 1 or lo % 10 in (0, 1)) and (hi == 100 or hi % 10 in (9, 0))


INTERVAL_ROUND = np.array([_is_round(c[0], c[-1]) for c in INTERVALS], dtype="int64")
LOG_N_ROUND = np.log(INTERVAL_ROUND.sum())
LOG_N_ARBITRARY = np.log(len(INTERVALS) - INTERVAL_ROUND.sum())


# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so prepare_observed collapses each set's intervals
# into <= 1000 groups (width x examples inside x round); the posterior is exact.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    group_round = pm.Data("group_round", np.zeros(N_GROUPS))  # (groups,) 1 = round boundaries
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # The single change: the examples' evidence is raised to this power.
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    # The single change: a response bias shifting the criterion on the log-odds.
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The new mechanism: log prior boost for intervals with round decade boundaries.
    round_boost = pm.Normal("round_boost", mu=0.0, sigma=1.5)
    log_interval_norm = pt.logaddexp(LOG_N_ARBITRARY, LOG_N_ROUND + round_boost)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + evidence_weight * rule_log_lik
    log_groups = pt.log1p(-p_mix) - log_interval_norm + round_boost * group_round[None, :] + group_log_count + evidence_weight * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    # Decision gain on the log-odds of the belief, plus a criterion shift (bias).
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
    log_lik = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1)
        group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside + N_BASE_GROUPS * INTERVAL_ROUND
        for g in np.unique(group):
            members = group == g
            log_count[i, g] = np.log(members.sum())
            log_lik[i, g] = ll[members][0]
            share[i, g] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    used = (log_count > -700.0).any(axis=0)  # drop groups empty for every set
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_count": log_count[:, used],
        "group_log_lik": log_lik[:, used],
        "group_in_concept": share[:, used][set_idx, :, tgt],
        "group_round": (np.arange(N_GROUPS) >= N_BASE_GROUPS).astype("float64")[used],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
