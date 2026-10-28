"""Tversky-similarity retrieval of stored number categories.

People do not infer which concept produced the examples; they retrieve
familiar number categories from memory (42 arithmetic rules with a kind-level
familiarity, and every stretch of 1-100 with a familiarity favouring moderate
widths) by how similar each stored category is to the example set under
Tversky's ratio model. Only categories containing every example are retrieved,
and sim = shown members / (shown members + beta * category members not shown).
Retrieval strength is familiarity * sim ** gamma (Luce choice over
categories), and "yes" is the retrieval-weighted share of categories
containing the target (with a guessed answer on an inferred share of
trials). Because similarity depends on proportions, not joint
probabilities, extra examples consistent with a small category do not tighten
generalization as the size principle would.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

SAMPLER_SETTINGS = {"target_accept": 0.9}

NUMBERS = np.arange(1, 101)


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, bounds


RULES, RULE_KINDS, BOUNDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in BOUNDS], dtype="float64")
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS], dtype="float64")
SIZES = RULE_MEMBERSHIP.sum(axis=1)
NEG = -1e4
# Intervals of the same width that contain every example have the same
# retrieval strength, so they are pooled into one group per width.
WIDTHS = np.arange(1.0, 101.0)
WD_OF_INTERVAL = INTERVAL_WIDTH.astype(int) - 1
WD_SIZES = 101.0 - WIDTHS
N_WD = len(WIDTHS)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_shown, rule_common, rule_excluded, group_log_count, group_mem = [], [], [], [], []
    for s in unique:
        ex = np.array(sorted(set(s)))
        n_shown.append(float(len(ex)))
        common = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1)
        rule_common.append(common)
        rule_excluded.append(np.where(common == len(ex), 0.0, NEG))
        ok = INTERVAL_MEMBERSHIP[:, ex - 1].sum(axis=1) == len(ex)
        g = WD_OF_INTERVAL[ok]
        counts = np.bincount(g, minlength=N_WD)
        mem = np.zeros((N_WD, 100))
        np.add.at(mem, g, INTERVAL_MEMBERSHIP[ok])
        has = counts > 0
        log_count = np.full(N_WD, NEG)
        log_count[has] = np.log(counts[has])
        mem[has] /= counts[has, None]
        group_log_count.append(log_count)
        group_mem.append(mem)
    return {
        "n_shown": np.array(n_shown),
        "rule_common": np.stack(rule_common),
        "rule_excluded": np.stack(rule_excluded),
        "group_log_count": np.stack(group_log_count),
        "group_membership": np.stack(group_mem),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array(targets, dtype="int64") - 1,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    n_shown = pm.Data("n_shown", np.ones(1))
    rule_common = pm.Data("rule_common", np.ones((1, N_RULES)))
    rule_excluded = pm.Data("rule_excluded", np.zeros((1, N_RULES)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_WD)))
    group_membership = pm.Data("group_membership", np.zeros((1, N_WD, 100)))
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Familiarity of stored categories: rules vs stretches, kinds of rule,
    # and preferred stretch widths (Erlang).
    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    # Tversky ratio-model weight of category members not shown (beta) and
    # retrieval sharpness gamma.
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)
    gamma = pm.LogNormal("gamma", mu=np.log(2.0), sigma=0.75)
    # Share of trials answered by guessing rather than from retrieval.
    lapse = pm.Beta("lapse", alpha=2.0, beta=8.0)

    def log_similarity(common, unshown):
        safe = pt.maximum(common, 1.0)
        return pt.log(safe) - pt.log(safe + beta * unshown)

    log_rule = pt.log(rule_share) + pt.log(pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS))
    log_wd = np.log(WIDTHS) - WIDTHS / width_scale
    log_wd = pt.log(1 - rule_share) + log_wd - pt.logsumexp(log_wd + np.log(WD_SIZES))

    rule_sim = log_similarity(rule_common, SIZES[None, :] - rule_common)
    rule_strength = log_rule[None, :] + gamma * rule_sim + rule_excluded
    group_sim = log_similarity(n_shown[:, None], pt.maximum(WIDTHS[None, :] - n_shown[:, None], 0.0))
    group_strength = log_wd[None, :] + group_log_count + gamma * group_sim
    strength = pt.concatenate([rule_strength, group_strength], axis=1)
    retrieval = pt.exp(strength - pt.logsumexp(strength, axis=1, keepdims=True))
    # Per set and target: retrieval-weighted share of categories containing it.
    in_category = pt.dot(retrieval[:, :N_RULES], RULE_MEMBERSHIP) + pt.batched_dot(
        retrieval[:, None, N_RULES:], group_membership
    )[:, 0, :]

    p_yes = pm.Deterministic("p_yes", pt.clip(0.5 * lapse + (1 - lapse) * in_category[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
