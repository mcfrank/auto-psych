"""Pooled rule familiarity on primacy-and-recency concept learning.

Refinement of serial_position_ends_inflated_size: people do the same Bayesian
concept learning over 42 rules and all intervals of 1..100, 10% noisy examples,
perceived concept sizes inflated by a constant, evidence tempered separately for
rules and ranges, the first and last examples weighted by their own inferred
multipliers, a decision gain and shifted criterion. The one change is to the
prior over rules: a rule's familiarity is its size-based plausibility (a power
of its size) times an idiosyncratic familiarity factor drawn from a wide
log-normal spread, instead of a Dirichlet(2) weight that pulls every rule's
weight toward the others. A few rules can therefore be strongly familiar and
many barely entertained, far more unevenly than the refined model allows.
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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()  # centred, for identifiability
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
SIZE_DIR = RULE_LOG_SIZE_C / np.linalg.norm(RULE_LOG_SIZE_C)  # unit size-trend direction
FAMILIARITY_SPREAD = 3.0
LOG_NOISE = np.log(EPSILON / 100)
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 4
ANCHOR = 0  # position of the anchoring example (first shown)
LAST = -1  # position of the recent example (last shown)

# Intervals of the same width containing the same number of examples (and the
# first and last examples or not) share a prior and a likelihood, so prepare_observed
# collapses each set's intervals into such groups; the posterior is exact.

with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    rule_anchor_in = pm.Data("rule_anchor_in", np.zeros((1, N_RULES)))  # (sets, rules)
    group_width = pm.Data("group_width", np.ones((1, N_GROUPS)))  # (sets, groups)
    group_inside = pm.Data("group_inside", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_anchor_in = pm.Data("group_anchor_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_last_in = pm.Data("group_last_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    rule_last_in = pm.Data("rule_last_in", np.zeros((1, N_RULES)))  # (sets, rules)
    multi = pm.Data("multi", np.zeros(1))  # (sets,) 1 if two or more examples
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_share = pm.Data("group_share", np.zeros((1, N_GROUPS, 1)))  # (sets, groups, asked targets)
    set_rules = pm.Data("set_rules", np.zeros((1, N_RULES, 1)))  # (sets, rules, asked targets)
    set_size = pm.Data("set_size", np.ones(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * t_max + asked-target slot
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    # The single change: idiosyncratic rule familiarity on the log scale, with a
    # wide fixed spread (Dirichlet(2) log-weights have sd ~0.8). The deviations
    # are kept orthogonal to the size trend so they cannot trade off with it.
    familiarity_z = pm.ZeroSumNormal("familiarity_z", sigma=1.0, shape=N_RULES)
    familiarity = FAMILIARITY_SPREAD * (familiarity_z - pt.dot(familiarity_z, SIZE_DIR) * SIZE_DIR)
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    extra = anchor_weight - 1.0
    # Perceived concept size = true size + inflation (as in the refined model).
    size_inflation = pm.LogNormal("size_inflation", mu=np.log(1.5), sigma=0.75)
    # The last example (sets of 2+) has its own multiplier.
    recency_weight = pm.LogNormal("recency_weight", mu=0.0, sigma=0.75)
    rec = ((recency_weight - 1.0) * multi)[:, None]

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    n = set_size[:, None]
    rule_in = in_lik(pt.as_tensor(RULE_SIZE))[None, :]
    rule_ev = rule_inside * rule_in + (n - rule_inside) * LOG_NOISE
    rule_ev = rule_ev + extra * (rule_anchor_in * rule_in + (1 - rule_anchor_in) * LOG_NOISE)
    rule_ev = rule_ev + rec * (rule_last_in * rule_in + (1 - rule_last_in) * LOG_NOISE)
    group_in = in_lik(group_width)
    group_ev = group_inside * group_in + (n - group_inside) * LOG_NOISE
    group_ev = group_ev + extra * (group_anchor_in * group_in + (1 - group_anchor_in) * LOG_NOISE)
    group_ev = group_ev + rec * (group_last_in * group_in + (1 - group_last_in) * LOG_NOISE)

    # Rule prior proportional to rule size ** size_exponent,
    # renormalised within the rules (0 = inflated_concept_size_principle).
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    log_rule_prior = familiarity + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Belief per (set, asked target): only the targets each set was asked about.
    cell_p = pt.batched_dot(posterior[:, :N_RULES], set_rules) + pt.batched_dot(posterior[:, N_RULES:], group_share)
    concept_p = pt.clip(cell_p.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)

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
    last_g = np.zeros((n_sets, N_GROUPS))
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        anchored = INTERVAL_MEMBERSHIP[:, ex[ANCHOR]].astype("int64")
        last = INTERVAL_MEMBERSHIP[:, ex[LAST]].astype("int64")
        group = (((INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside) * 2 + anchored) * 2 + last
        for j, g in enumerate(np.unique(group)):  # compact: occupied groups only
            members = group == g
            first = np.flatnonzero(members)[0]
            log_count[i, j] = np.log(members.sum())
            width[i, j] = INTERVAL_WIDTH[first]
            inside_g[i, j] = inside[first]
            anchor_g[i, j] = anchored[first]
            last_g[i, j] = last[first]
            share[i, j] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    used = int((log_count > -700.0).sum(axis=1).max())
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    asked = [set() for _ in range(n_sets)]
    for si, t in zip(set_idx.tolist(), tgt.tolist()):
        asked[si].add(t)
    asked = [sorted(a) for a in asked]
    t_max = max(len(a) for a in asked)
    pad = [a + [a[0]] * (t_max - len(a)) for a in asked]
    pos = [{t: k for k, t in enumerate(a)} for a in asked]
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_in": np.stack([RULE_MEMBERSHIP[:, s[ANCHOR] - 1] for s in unique]),
        "group_width": width[:, :used],
        "group_inside": inside_g[:, :used],
        "group_anchor_in": anchor_g[:, :used],
        "group_last_in": last_g[:, :used],
        "rule_last_in": np.stack([RULE_MEMBERSHIP[:, s[LAST] - 1] for s in unique]),
        "multi": np.array([1.0 if len(s) > 1 else 0.0 for s in unique]),
        "group_log_count": log_count[:, :used],
        "group_share": np.stack([share[i][:used][:, pad[i]] for i in range(n_sets)]),
        "set_rules": np.stack([RULE_MEMBERSHIP[:, pad[i]] for i in range(n_sets)]),
        "set_size": np.array([float(len(s)) for s in unique]),
        "row_cell": np.array([si * t_max + pos[si][t] for si, t in zip(set_idx.tolist(), tgt.tolist())], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
