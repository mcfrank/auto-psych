"""Disjunctions of two familiar rules added to the concept space.

People do the same Bayesian concept learning as inflated_size_familiar_rule_prior
(42 weighted rules with a size-dependent prior and all intervals of 1..100, 10%
noisy examples, perceived sizes inflated by a constant, evidence tempered
separately for rules and ranges, the first example weighted by an inferred
multiplier, a decision gain and shifted criterion), but their concept space also
contains unions of two different rules ("ends in 2 or ends in 7", "multiples of
8 or multiples of 9"). The unions share an inferred, a-priori small fraction
p_union of the rule belief, spread equally over the unions (a union is not
weighted by its component rules' fitted priors). A union is judged like a
rule: size principle on its (inflated) size, rule evidence weight. p_union = 0
recovers inflated_size_familiar_rule_prior.
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


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()  # centred, for identifiability
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
LOG_NOISE = np.log(EPSILON / 100)
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
ANCHOR = 0  # position of the anchoring example (first shown)

# Unions of two different rules (a union identical to one of its rules, e.g.
# "powers of 4 or even", is dropped: it is that rule).
_pairs = []
for i, j in itertools.combinations(range(N_RULES), 2):
    u = np.maximum(RULE_MEMBERSHIP[i], RULE_MEMBERSHIP[j])
    if u.sum() > max(RULE_SIZE[i], RULE_SIZE[j]):
        _pairs.append((i, j))
PAIR_I = np.array([p[0] for p in _pairs], dtype="int64")
PAIR_J = np.array([p[1] for p in _pairs], dtype="int64")
UNION_MEMBERSHIP = np.maximum(RULE_MEMBERSHIP[PAIR_I], RULE_MEMBERSHIP[PAIR_J])
UNION_SIZE = UNION_MEMBERSHIP.sum(axis=1)
N_UNIONS = len(_pairs)
LOG_UNION_PRIOR = -np.log(N_UNIONS)
ARB_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, UNION_MEMBERSHIP])  # rules then unions

with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    rule_anchor_in = pm.Data("rule_anchor_in", np.zeros((1, N_RULES)))  # (sets, rules)
    union_inside = pm.Data("union_inside", np.zeros((1, N_UNIONS)))  # (sets, unions)
    union_anchor_in = pm.Data("union_anchor_in", np.zeros((1, N_UNIONS)))  # (sets, unions)
    group_width = pm.Data("group_width", np.ones((1, N_GROUPS)))  # (sets, groups)
    group_inside = pm.Data("group_inside", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_anchor_in = pm.Data("group_anchor_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    # Each target's share of an interval group's members (sets, groups, targets).
    group_share = pm.Data("group_share", np.zeros((1, N_GROUPS, 100)))
    set_size = pm.Data("set_size", np.ones(1))  # (sets,)
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
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # The single change: the share of rule belief given to two-rule unions.
    p_union = pm.Beta("p_union", 2.0, 20.0)

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    def evidence(inside, anchor_in, lik):
        ev = inside * lik + (n - inside) * LOG_NOISE
        return ev + extra * (anchor_in * lik + (1 - anchor_in) * LOG_NOISE)

    n = set_size[:, None]
    rule_ev = evidence(rule_inside, rule_anchor_in, in_lik(pt.as_tensor(RULE_SIZE))[None, :])
    union_ev = evidence(union_inside, union_anchor_in, in_lik(pt.as_tensor(UNION_SIZE))[None, :])
    group_ev = evidence(group_inside, group_anchor_in, in_lik(group_width))

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    log_rules = pt.log(p_mix) + pt.log1p(-p_union) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_unions = pt.log(p_mix) + pt.log(p_union) + LOG_UNION_PRIOR + rule_evidence_weight * union_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_unions, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    n_arb = N_RULES + N_UNIONS
    # Belief that each number 1..100 is in the concept, per set (sets, 100).
    grid = pt.dot(posterior[:, :n_arb], pt.as_tensor(ARB_MEMBERSHIP))
    grid = grid + pt.batched_dot(posterior[:, None, n_arb:], group_share)[:, 0, :]
    concept_p = pt.clip(grid.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)

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
    inside_g = np.zeros((n_sets, N_GROUPS))
    anchor_g = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        anchored = INTERVAL_MEMBERSHIP[:, ex[ANCHOR]].astype("int64")
        group = ((INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside) * 2 + anchored
        for j, g in enumerate(np.unique(group)):  # compact: occupied groups only
            members = group == g
            first = np.flatnonzero(members)[0]
            log_count[i, j] = np.log(members.sum())
            width[i, j] = INTERVAL_WIDTH[first]
            inside_g[i, j] = inside[first]
            anchor_g[i, j] = anchored[first]
            share[i, j] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    used = int((log_count > -700.0).sum(axis=1).max())
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_in": np.stack([RULE_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "union_inside": np.stack([UNION_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "union_anchor_in": np.stack([UNION_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "group_width": width[:, :used],
        "group_inside": inside_g[:, :used],
        "group_anchor_in": anchor_g[:, :used],
        "group_log_count": log_count[:, :used],
        "group_share": share[:, :used, :],
        "set_size": np.array([float(len(s)) for s in unique]),
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
