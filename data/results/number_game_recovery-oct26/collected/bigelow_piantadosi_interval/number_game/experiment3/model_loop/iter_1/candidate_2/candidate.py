"""Intermittent access to arithmetic rules; numeral-surface reasoning otherwise.

People do not always work out the arithmetic structure of the examples: on an
inferred fraction of judgments (1 - arithmetic_access) they reason only with
properties readable straight off the written numerals and their magnitude
(odd/even, final digit, digits contained, multiples of 5 and 10, numeric
ranges); on the rest they also consider computed rules (powers, squares, cubes,
primes, multiples of 3, 4, 6, 7, 8, 9, 11, 12). Within whichever hypothesis
space they hold they do the Bayesian concept learning of the best model
(inflated_size_familiar_rule_prior: weighted rules with a size-dependent prior,
all intervals, sizes inflated by a constant, tempered rule and range evidence,
first-example weighting, decision gain and criterion), the rule prior being
renormalised over the rules they have in mind. Surface-only judgments keep
generalizing to other even numbers or shared final digits however many
power-of-two examples are shown; arithmetic_access = 1 recovers the best model.
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
LOG_NOISE = np.log(EPSILON / 100)
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1) * 2
ANCHOR = 0
# Rules readable from the numeral itself: multiples of 5 and 10, ends in d,
# contains d, even, odd. All others need arithmetic.
SURFACE = np.zeros(N_RULES, dtype=bool)
SURFACE[[11, 16]] = True
SURFACE[19:37] = True
SURFACE[[38, 39]] = True
SURFACE_LOG_MASK = np.where(SURFACE, 0.0, -1e4)  # position of the anchoring example (first shown)

# Intervals of the same width containing the same number of examples (and the
# first example or not) share a prior and a likelihood, so prepare_observed
# collapses each set's intervals into such groups; the posterior is exact.

with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    rule_anchor_in = pm.Data("rule_anchor_in", np.zeros((1, N_RULES)))  # (sets, rules)
    group_width = pm.Data("group_width", np.ones((1, N_GROUPS)))  # (sets, groups)
    group_inside = pm.Data("group_inside", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_anchor_in = pm.Data("group_anchor_in", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
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
    # Perceived concept size = true size + inflation (as in the refined model).
    size_inflation = pm.HalfNormal("size_inflation", sigma=10.0)

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    n = set_size[:, None]
    rule_in = in_lik(pt.as_tensor(RULE_SIZE))[None, :]
    rule_ev = rule_inside * rule_in + (n - rule_inside) * LOG_NOISE
    rule_ev = rule_ev + extra * (rule_anchor_in * rule_in + (1 - rule_anchor_in) * LOG_NOISE)
    group_in = in_lik(group_width)
    group_ev = group_inside * group_in + (n - group_inside) * LOG_NOISE
    group_ev = group_ev + extra * (group_anchor_in * group_in + (1 - group_anchor_in) * LOG_NOISE)

    # The single change: rule prior proportional to rule size ** size_exponent,
    # renormalised within the rules (0 = inflated_concept_size_principle).
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + range_evidence_weight * group_ev
    rule_mat = pt.as_tensor(RULE_MEMBERSHIP)
    gain = pt.exp(log_gain)

    def answer(log_prior):
        log_rules = pt.log(p_mix) + log_prior[None, :] + rule_evidence_weight * rule_ev
        log_joint = pt.concatenate([log_rules, log_groups], axis=1)
        posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
        rule_part = (posterior[:, :N_RULES] @ rule_mat).reshape((-1,))[row_cell]
        interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
        concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)
        belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
        return pm.math.sigmoid(gain * belief_logit + bias)

    # The single change: arithmetic rules are in mind only on a fraction of
    # judgments; otherwise the prior is renormalised over the surface rules.
    arithmetic_access = pm.Beta("arithmetic_access", 2.0, 2.0)
    surface_prior = log_rule_prior + SURFACE_LOG_MASK
    surface_prior = surface_prior - pt.logsumexp(surface_prior)
    p_mixed = arithmetic_access * answer(log_rule_prior) + (1 - arithmetic_access) * answer(surface_prior)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_mixed, 1e-6, 1 - 1e-6))
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
        "group_width": width[:, :used],
        "group_inside": inside_g[:, :used],
        "group_anchor_in": anchor_g[:, :used],
        "group_log_count": log_count[:, :used],
        "group_in_concept": share[set_idx, :, tgt][:, :used],
        "set_size": np.array([float(len(s)) for s in unique]),
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
