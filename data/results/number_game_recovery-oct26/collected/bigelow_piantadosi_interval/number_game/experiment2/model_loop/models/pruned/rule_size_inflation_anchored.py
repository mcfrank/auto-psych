"""Subjective rule-size inflation on top of first-example evidence weighting.

Refinement of first_example_evidence_weighting: people do the same Bayesian
concept learning (42 weighted rules and all intervals of 1..100, size principle
with evidence tempered separately for rules and ranges, 10% noisy examples, the
first example's evidence weighted by an inferred multiplier, a decision gain
and shifted criterion), but they judge each rule's size as if it had an
inferred number of extra members beyond its true count in 1..100. The one
change is this pseudo-count on rule size, which shrinks the size-principle
advantage of tiny rules (powers of 2 or 4) over broad ones (even, ends in 4,
multiples of 8), more so the more examples are seen; zero recovers the refined
model.
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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
LOG_NOISE = np.log(EPSILON / 100)
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
ANCHOR = 0  # position of the anchoring example (first shown)

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so prepare_observed collapses each set's intervals
# into groups (width x examples inside x contains the first example), keeping
# only the occupied groups; the posterior is exact.

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules) examples in rule
    rule_anchor_in = pm.Data("rule_anchor_in", np.zeros((1, N_RULES)))  # (sets, rules) first example in rule
    set_size = pm.Data("set_size", np.zeros(1))  # (sets,)
    group_anchor_lik = pm.Data("group_anchor_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
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
    # The single change: every rule is judged as if it had extra_members more
    # members than it truly has (0 = exact counting, the refined model).
    extra_members = pm.HalfNormal("extra_members", sigma=10.0)

    log_member = pt.log((1 - EPSILON) / (pt.as_tensor(RULE_SIZE) + extra_members) + EPSILON / 100)
    n_member = rule_n_in + extra * rule_anchor_in
    n_noise = set_size[:, None] + extra - n_member
    rule_ev = n_member * log_member[None, :] + n_noise * LOG_NOISE
    group_ev = group_log_lik + extra * group_anchor_lik
    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_evidence_weight * rule_ev
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
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_in": np.stack([RULE_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique]),
        "group_anchor_lik": anchor_lik,
        "group_log_count": log_count,
        "group_log_lik": log_lik,
        "group_in_concept": share[set_idx, :, tgt][:, :used],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
