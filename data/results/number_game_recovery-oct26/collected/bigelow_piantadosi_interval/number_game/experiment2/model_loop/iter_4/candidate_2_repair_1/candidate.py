"""Graded similarity beyond the edges of believed ranges.

People infer the program's concept by the same Bayesian concept learning as
rule_size_prior_anchored_evidence (42 weighted rules with a size-dependent
prior and all intervals of 1..100, size principle with evidence tempered
separately for rules and ranges, 10% noisy examples, a first example weighed by
an inferred multiplier, a decision gain and shifted criterion). The one change
is how they generalize from a believed numeric range: not by crisp membership
but by similarity on the number line. A target inside the range counts fully;
one outside counts as edge_q ** d, d its distance to the range's nearest edge,
with edge_q the inferred share of similarity kept per step. Arithmetic rules are
still judged by exact membership; edge_q = 0 recovers the refined model.
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
MAX_DIST = 20  # similarity beyond this distance from a range edge is treated as 0
INTERVAL_LO = np.array([c[0] for c in INTERVALS])
INTERVAL_HI = np.array([c[-1] for c in INTERVALS])

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
    # Sparse (cell, group, distance) table: the share of a group's intervals lying at
    # each distance from the cell's target, with flat indices into the group posterior.
    entry_group = pm.Data("entry_group", np.zeros(1, dtype="int64"))  # set * groups + group
    entry_dist = pm.Data("entry_dist", np.zeros(1, dtype="int64"))
    entry_share = pm.Data("entry_share", np.zeros(1))
    entry_cell = pm.Data("entry_cell", np.zeros(1, dtype="int64"))
    cell_rule = pm.Data("cell_rule", np.zeros(1, dtype="int64"))  # -> set index * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # trial -> distinct (set, target)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # Evidence tempered separately for rules and for ranges.
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    # Response bias shifting the criterion on the log-odds (as in the incumbent).
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The first example's evidence counts anchor_weight times
    # as much as a later example's (1 = no anchoring).
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    extra = anchor_weight - 1.0
    # Rule prior proportional to rule size ** size_exponent,
    # renormalised within the rules (as in the refined model).
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # The mechanism: share of similarity kept per step beyond a believed range's edge.
    edge_q = pm.Beta("edge_q", 1.5, 5.0)
    q_powers = edge_q ** pt.as_tensor(np.arange(MAX_DIST + 1, dtype="float64"))  # (distances,)
    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    rule_ev = rule_log_lik + extra * rule_anchor_lik
    group_ev = group_log_lik + extra * group_anchor_lik
    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[cell_rule]
    entry_mass = posterior[:, N_RULES:].reshape((-1,))[entry_group] * entry_share * q_powers[entry_dist]
    interval_part = pt.zeros(cell_rule.shape[0])[entry_cell].inc(entry_mass)  # graded range membership
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    # Decision gain on the log-odds of the belief, plus a criterion shift (bias).
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
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    cells, row_cell = np.unique(set_idx * 100 + tgt, return_inverse=True)
    cell_set, cell_tgt = cells // 100, cells % 100
    # Distance of every number (0-based) to every interval, capped past MAX_DIST.
    dist = np.maximum(INTERVAL_LO[None, :] - NUMBERS[:, None], 0) + np.maximum(NUMBERS[:, None] - INTERVAL_HI[None, :], 0)
    dist = np.minimum(dist, MAX_DIST + 1)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    log_lik = np.zeros((n_sets, N_GROUPS))
    anchor_lik = np.zeros((n_sets, N_GROUPS))
    groups = []
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1)
        a_ll = EXAMPLE_LOG_LIK[N_RULES:, ex[ANCHOR]]
        anchored = INTERVAL_MEMBERSHIP[:, ex[ANCHOR]].astype("int64")
        group = ((INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside) * 2 + anchored
        uniq, compact = np.unique(group, return_inverse=True)  # occupied groups only
        groups.append(compact.reshape(-1))
        for j, g in enumerate(uniq):
            members = group == g
            log_count[i, j] = np.log(members.sum())
            log_lik[i, j] = ll[members][0]
            anchor_lik[i, j] = a_ll[members][0]
    used = int((log_count > -700.0).sum(axis=1).max())
    e_group, e_dist, e_share, e_cell = [], [], [], []
    for c, (i, t) in enumerate(zip(cell_set, cell_tgt)):
        g = groups[i]
        table = np.zeros((used, MAX_DIST + 2))
        np.add.at(table, (g, dist[t]), 1.0)
        table /= np.maximum(table.sum(axis=1, keepdims=True), 1.0)
        gg, dd = np.nonzero(table[:, : MAX_DIST + 1])
        e_group.append(i * used + gg)
        e_dist.append(dd)
        e_share.append(table[gg, dd])
        e_cell.append(np.full(len(gg), c))
    return {
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, s[ANCHOR] - 1] for s in unique]),
        "group_anchor_lik": anchor_lik[:, :used],
        "group_log_count": log_count[:, :used],
        "group_log_lik": log_lik[:, :used],
        "entry_group": np.concatenate(e_group).astype("int64"),
        "entry_dist": np.concatenate(e_dist).astype("int64"),
        "entry_share": np.concatenate(e_share),
        "entry_cell": np.concatenate(e_cell).astype("int64"),
        "cell_rule": cells.astype("int64"),
        "row_cell": row_cell.astype("int64").reshape(-1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
