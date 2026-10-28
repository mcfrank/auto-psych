"""Inflated concept sizes with final-digit union concepts.

Refinement of inflated_concept_size_principle: people do the same Bayesian
concept learning (42 weighted rules and all intervals of 1..100, 10% noisy
examples, perceived concept size = true size + an inferred inflation, evidence
tempered separately for rules and ranges, the first example weighted by an
inferred multiplier, a decision gain and shifted criterion). The one change:
because people see numbers as written digits, their concepts also include
"ends in any of these digits" -- every union of 2, 3 or 4 final digits (375
concepts) -- sharing an inferred slice `union_share` of the rule prior (split
equally across arities, then equally within one). After 2, 8 the union "ends
in 2 or 8" is smaller than "even" and less exotic than "powers of 2", so people
endorse 62 or 88 more than 16; zero share recovers inflated_concept_size_principle.
"""

import itertools

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


def digit_unions():
    """Every union of 2, 3 or 4 final-digit classes (membership, arity)."""
    members, arity = [], []
    for k in (2, 3, 4):
        for digits in itertools.combinations(range(10), k):
            members.append(np.isin(NUMBERS % 10, digits))
            arity.append(k)
    return np.array(members, dtype="float64"), np.array(arity, dtype="int64")


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
UNION_MEMBERSHIP, UNION_ARITY = digit_unions()
# Within the union slice: 1/3 per arity, equal within an arity.
UNION_LOG_PRIOR = np.log(1.0 / 3.0) - np.log(np.bincount(UNION_ARITY)[UNION_ARITY])
LOG_NOISE = np.log(EPSILON / 100)
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
N_UGROUPS = 3 * (MAX_SET + 1) * 2
ANCHOR = 0  # position of the anchoring example (first shown)

# Intervals of the same width (and unions of the same arity) containing the same
# number of examples, and the first example or not, share a prior and a
# likelihood, so prepare_observed collapses them into groups; the posterior is
# exact.

with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    rule_anchor_in = pm.Data("rule_anchor_in", np.zeros((1, N_RULES)))  # (sets, rules)
    group_width = pm.Data("group_width", np.ones((1, N_GROUPS)))  # (sets, groups)
    group_inside = pm.Data("group_inside", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_anchor_in = pm.Data("group_anchor_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    ugroup_size = pm.Data("ugroup_size", np.full((1, N_UGROUPS), 20.0))  # (sets, union groups)
    ugroup_inside = pm.Data("ugroup_inside", np.zeros((1, N_UGROUPS)))
    ugroup_anchor_in = pm.Data("ugroup_anchor_in", np.zeros((1, N_UGROUPS)))
    ugroup_log_prior = pm.Data("ugroup_log_prior", np.zeros((1, N_UGROUPS)))  # within the slice, incl. count
    ugroup_in_concept = pm.Data("ugroup_in_concept", np.zeros((1, N_UGROUPS)))  # (trials, union groups)
    set_size = pm.Data("set_size", np.ones(1))  # (sets,)
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
    size_inflation = pm.HalfNormal("size_inflation", sigma=10.0)
    # The single change: share of the rule prior on final-digit unions.
    union_share = pm.Beta("union_share", 1.0, 9.0)

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    def evidence(inside, anchor_in, lik):
        ev = inside * lik + (n - inside) * LOG_NOISE
        return ev + extra * (anchor_in * lik + (1 - anchor_in) * LOG_NOISE)

    n = set_size[:, None]
    rule_ev = evidence(rule_inside, rule_anchor_in, in_lik(pt.as_tensor(RULE_SIZE))[None, :])
    group_ev = evidence(group_inside, group_anchor_in, in_lik(group_width))
    union_ev = evidence(ugroup_inside, ugroup_anchor_in, in_lik(ugroup_size))

    log_p_rules = pt.log(p_mix)
    log_rules = log_p_rules + pt.log1p(-union_share) + pt.log(rule_weights)[None, :] + rule_evidence_weight * rule_ev
    log_unions = log_p_rules + pt.log(union_share) + ugroup_log_prior + rule_evidence_weight * union_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_unions, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    n_ug = ugroup_size.shape[1]
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    union_part = (posterior[:, N_RULES : N_RULES + n_ug][row_set] * ugroup_in_concept).sum(axis=1)
    interval_part = (posterior[:, N_RULES + n_ug :][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + union_part + interval_part, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _collapse(member, size, log_prior, ex, n_slots):
    """Group concepts by (size, examples inside, anchor inside)."""
    inside = member[:, ex].sum(axis=1).astype("int64")
    anchored = member[:, ex[ANCHOR]].astype("int64")
    key = ((size - 1) * (MAX_SET + 1) + inside) * 2 + anchored
    out_lp = np.full(n_slots, -700.0)  # empty groups get no mass
    out_size = np.full(n_slots, 20.0)
    out_in = np.zeros(n_slots)
    out_anchor = np.zeros(n_slots)
    share = np.zeros((n_slots, 100))
    for j, g in enumerate(np.unique(key)):
        members = key == g
        first = np.flatnonzero(members)[0]
        out_lp[j] = np.log(np.exp(log_prior[members]).sum())
        out_size[j] = size[first]
        out_in[j] = inside[first]
        out_anchor[j] = anchored[first]
        share[j] = member[members].mean(axis=0)
    return out_lp, out_size, out_in, out_anchor, share


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    union_size = (10 * UNION_ARITY).astype("int64")
    zero_lp = np.zeros(len(INTERVALS))
    g = [_collapse(INTERVAL_MEMBERSHIP, INTERVAL_WIDTH, zero_lp, np.array(s) - 1, N_GROUPS) for s in unique]
    u = [_collapse(UNION_MEMBERSHIP, union_size, UNION_LOG_PRIOR, np.array(s) - 1, N_UGROUPS) for s in unique]
    used = int(max((x[0] > -700.0).sum() for x in g))
    used_u = int(max((x[0] > -700.0).sum() for x in u))
    stack = lambda parts, k, m: np.stack([p[k][:m] for p in parts])
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    g_share = np.stack([p[4][:used] for p in g])
    u_share = np.stack([p[4][:used_u] for p in u])
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_in": np.stack([RULE_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "group_width": stack(g, 1, used),
        "group_inside": stack(g, 2, used),
        "group_anchor_in": stack(g, 3, used),
        "group_log_count": stack(g, 0, used),
        "group_in_concept": g_share[set_idx, :, tgt],
        "ugroup_size": stack(u, 1, used_u),
        "ugroup_inside": stack(u, 2, used_u),
        "ugroup_anchor_in": stack(u, 3, used_u),
        "ugroup_log_prior": stack(u, 0, used_u),
        "ugroup_in_concept": u_share[set_idx, :, tgt],
        "set_size": np.array([float(len(s)) for s in unique]),
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
