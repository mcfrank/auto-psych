"""Range-bounded rule concepts with split rule/range evidence tempering.

Refinement of range_bounded_rule_concepts: people do Bayesian concept learning
over 42 weighted rules, their range-bounded runs (an inferred share
`bounded_share` of each rule's prior, divided equally among its runs) and all
5,050 intervals of 1..100 (size principle, 10% noisy examples), answering with
p_yes = sigmoid(gain * logit(P(target in concept | D)) + bias). The single
change, taken from split_rule_range_tempering: the examples' evidence is
tempered by one inferred weight for arithmetic concepts (rules and their
bounded runs) and another for plain ranges. Equal weights recover
range_bounded_rule_concepts.
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
RULE_MEMBERS = [np.array(sorted(int(x) for x in r)) for r in RULES]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)
EMPTY = -700.0

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, and so do a rule's bounded runs of the same length
# that cover the examples; prepare_observed collapses both into groups, so the
# posterior is exact over the enumerated concepts.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (cells, groups)
    run_rule = pm.Data("run_rule", np.zeros((1, 1), dtype="int64"))  # (sets, run groups)
    run_log_count = pm.Data("run_log_count", np.zeros((1, 1)))  # log(#runs / #runs of rule)
    run_log_lik = pm.Data("run_log_lik", np.zeros((1, 1)))
    run_in_concept = pm.Data("run_in_concept", np.zeros((1, 1)))  # (cells, run groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # distinct (set, target) cells
    cell_code = pm.Data("cell_code", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # each trial's cell
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # The single change: separate evidence tempering for rules (and their runs) and for ranges.
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # Share of each rule's prior given to its range-bounded runs.
    bounded_share = pm.Beta("bounded_share", 2.0, 5.0)

    log_w = pt.log(rule_weights)
    log_rules = (pt.log(p_mix) + pt.log1p(-bounded_share) + log_w[None, :]
                 + rule_evidence_weight * rule_log_lik)
    log_runs = (pt.log(p_mix) + pt.log(bounded_share) + log_w[run_rule]
                + run_log_count + rule_evidence_weight * run_log_lik)
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups, log_runs], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # P(target in concept) per distinct (set, target) cell, then read off per trial.
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[cell_code]
    in_concept = pt.concatenate([group_in_concept, run_in_concept], axis=1)
    grouped_part = (posterior[:, N_RULES:][cell_set] * in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + grouped_part, 1e-6, 1 - 1e-6)[row_cell]

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def bounded_run_groups(examples):
    """Groups (rule, run length) of the bounded runs of each rule that covers every example.

    Returns a list of (rule, log share of rule's runs, log-likelihood, P(target in run)).
    Bounded runs of a rule some example breaks, or runs leaving out an example, are
    omitted: their mass is negligible next to the unbounded rule and the intervals.
    """
    groups = []
    for r, members in enumerate(RULE_MEMBERS):
        m = len(members)
        pos = np.searchsorted(members, examples)
        inside = (pos < m) & (members[np.minimum(pos, m - 1)] == examples)
        if not inside.all():
            continue
        lo, hi = pos[inside].min(), pos[inside].max()
        n_runs_rule = m * (m + 1) / 2
        for size in range(hi - lo + 1, m):  # size m is the unbounded rule itself
            starts = np.arange(max(0, hi - size + 1), min(lo, m - size) + 1)
            if len(starts) == 0:
                continue
            cover = np.zeros(100)
            for s in starts:
                cover[members[s:s + size] - 1] += 1
            ll = len(examples) * np.log((1 - EPSILON) / size + EPSILON / 100)
            groups.append((r, np.log(len(starts) / n_runs_rule), ll, cover / len(starts)))
    return groups


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), EMPTY)
    log_lik = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))
    run_groups = []
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
        run_groups.append(bounded_run_groups(np.array(sorted(set(s)))))
    k = max(1, max(len(g) for g in run_groups))
    r_rule = np.zeros((n_sets, k), dtype="int64")
    r_count = np.full((n_sets, k), EMPTY)
    r_lik = np.zeros((n_sets, k))
    r_share = np.zeros((n_sets, k, 100))
    for i, gs in enumerate(run_groups):
        for j, (r, lc, ll, cov) in enumerate(gs):
            r_rule[i, j], r_count[i, j], r_lik[i, j], r_share[i, j] = r, lc, ll, cov
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    codes, row_cell = np.unique(set_idx * 100 + tgt, return_inverse=True)
    c_set, c_tgt = codes // 100, codes % 100
    return {
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_count": log_count,
        "group_log_lik": log_lik,
        "group_in_concept": share[c_set, :, c_tgt],
        "run_rule": r_rule,
        "run_log_count": r_count,
        "run_log_lik": r_lik,
        "run_in_concept": r_share[c_set, :, c_tgt],
        "cell_set": c_set,
        "cell_code": codes,
        "row_cell": row_cell,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
