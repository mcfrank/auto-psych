"""Surprise-weighted sequential evidence.

People read the examples one at a time, and how much each new example teaches
them depends on how surprising it is given the guess they have formed so far:
an example their current belief predicted well is treated as old news and
weighted less (or, under confirmation bias, more), while an unexpected example
moves their belief more (or less), by an inferred surprise sensitivity.
Everything else is Bayesian concept learning over 42 weighted rules (prior
rising as a power of rule size) and all intervals of 1..100, 10% noisy
examples, perceived concept sizes inflated by a constant, evidence tempered
separately for rules and ranges, a re-weighted first example, a decision gain
and a shifted criterion. Zero sensitivity recovers order-blind learning.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
LOG_NOISE = np.log(EPSILON / 100)
SURPRISE_CAP = 4.0  # soft cap on |log(100 * predictive probability)|

# Intervals of the same width containing the same subset of the examples share
# a prior and every step's likelihood, so prepare_observed collapses each set's
# intervals into such groups; the sequential posterior is exact.

with pm.Model() as model:
    rule_in_k = pm.Data("rule_in_k", np.zeros((1, N_RULES, MAX_SET)))  # (sets, rules, positions)
    group_in_k = pm.Data("group_in_k", np.zeros((1, 1, MAX_SET)))  # (sets, groups, positions)
    group_width = pm.Data("group_width", np.ones((1, 1)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))
    pos_mask = pm.Data("pos_mask", np.zeros((1, MAX_SET)))  # (sets, positions) example present
    group_share = pm.Data("group_share", np.zeros((1, 1, 1)))  # (sets, groups, asked targets)
    set_rules = pm.Data("set_rules", np.zeros((1, N_RULES, 1)))  # (sets, rules, asked targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_inflation = pm.LogNormal("size_inflation", mu=np.log(2.0), sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # The mechanism: an example's weight scales as (predictive prob) ** -sensitivity.
    surprise_sensitivity = pm.Normal("surprise_sensitivity", mu=0.0, sigma=0.5)

    def in_lik(size):
        return pt.log((1 - EPSILON) / (size + size_inflation) + EPSILON / 100)

    rule_in = in_lik(pt.as_tensor(RULE_SIZE))[None, :, None]
    group_in = in_lik(group_width)[:, :, None]
    # Untempered log-likelihood of each example under each concept: (sets, concepts, positions).
    rule_ll = rule_in_k * rule_in + (1 - rule_in_k) * LOG_NOISE
    group_ll = group_in_k * group_in + (1 - group_in_k) * LOG_NOISE
    concept_ll = pt.concatenate([rule_ll, group_ll], axis=1)
    concept_in = pt.concatenate([rule_in_k, group_in_k], axis=1)
    # Probability of an in-concept example, without the noise term: (sets, concepts).
    member_p = pt.concatenate(
        [pt.zeros_like(group_width[:, :N_RULES]) + (1 - EPSILON) / (pt.as_tensor(RULE_SIZE) + size_inflation),
         (1 - EPSILON) / (group_width + size_inflation)], axis=1
    )
    temper = pt.concatenate(
        [pt.fill(rule_ll[0, :, 0], rule_evidence_weight), pt.fill(group_ll[0, :, 0], range_evidence_weight)]
    )[None, :]

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    log_rules = pt.log(p_mix) + pt.zeros_like(rule_ll[:, :, 0]) + log_rule_prior[None, :]
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)

    log_joint = log_joint + anchor_weight * temper * concept_ll[:, :, 0]
    # Sets are sorted by size (largest first), so step k only touches the first n_k sets.
    for k in range(1, MAX_SET):
        n_k = pt.sum(pos_mask[:, k]).astype("int64")
        lj = log_joint[:n_k]
        post = pt.exp(lj - pt.logsumexp(lj, axis=1, keepdims=True))
        pred = pt.sum(post * concept_in[:n_k, :, k] * member_p[:n_k], axis=1) + EPSILON / 100
        surprise = SURPRISE_CAP * pt.tanh(pt.log(100.0 * pred) / SURPRISE_CAP)
        weight = pt.exp(-surprise_sensitivity * surprise)
        log_joint = pt.set_subtensor(log_joint[:n_k], lj + weight[:, None] * temper * concept_ll[:n_k, :, k])

    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
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
    unique = sorted(dict.fromkeys(sets), key=len, reverse=True)  # largest sets first
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    per_set = []
    for s in unique:
        ex = np.array(s) - 1
        bits = np.zeros(len(INTERVALS), dtype="int64")
        for k, e in enumerate(ex):
            bits += INTERVAL_MEMBERSHIP[:, e].astype("int64") << k
        key = (INTERVAL_WIDTH - 1) * 16 + bits
        groups = []
        for g in np.unique(key):
            members = key == g
            first = np.flatnonzero(members)[0]
            inside = np.zeros(MAX_SET)
            inside[: len(s)] = INTERVAL_MEMBERSHIP[first, ex]
            groups.append((np.log(members.sum()), INTERVAL_WIDTH[first], inside, INTERVAL_MEMBERSHIP[members].mean(axis=0)))
        per_set.append(groups)
    n_groups = max(len(g) for g in per_set)
    log_count = np.full((n_sets, n_groups), -700.0)  # padding groups get no mass
    width = np.ones((n_sets, n_groups))
    g_in = np.zeros((n_sets, n_groups, MAX_SET))
    share = np.zeros((n_sets, n_groups, 100))
    r_in = np.zeros((n_sets, N_RULES, MAX_SET))
    mask = np.zeros((n_sets, MAX_SET))
    for i, (s, groups) in enumerate(zip(unique, per_set)):
        for j, (lc, w, inside, sh) in enumerate(groups):
            log_count[i, j], width[i, j], g_in[i, j], share[i, j] = lc, w, inside, sh
        r_in[i, :, : len(s)] = RULE_MEMBERSHIP[:, np.array(s) - 1]
        mask[i, : len(s)] = 1.0
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
        "rule_in_k": r_in,
        "group_in_k": g_in,
        "group_width": width,
        "group_log_count": log_count,
        "pos_mask": mask,
        "group_share": np.stack([share[i][:, pad[i]] for i in range(n_sets)]),
        "set_rules": np.stack([RULE_MEMBERSHIP[:, pad[i]] for i in range(n_sets)]),
        "row_cell": np.array([si * t_max + pos[si][t] for si, t in zip(set_idx.tolist(), tgt.tolist())], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
