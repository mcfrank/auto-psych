"""Uncertainty-gated guessing over simple concept learning.

People infer the program's concept by plain Bayesian concept learning over 42
arithmetic rules (equal prior weight within the family) and all 5,050 intervals
of 1..100 (equally likely), a rules-vs-ranges prior share p_mix, examples sampled
from the concept (size principle, 10% stray numbers) with an inferred evidence
weight. The decision rule is the hypothesis: how often people abandon their
belief and guess (with an inferred yes-leaning) depends on how uncertain that
belief is -- the normalised entropy of the posterior over concepts --
guess_rate = sigmoid(guess_base + guess_slope * (H / H_max)). Obvious-rule sets
(low entropy) give answers from the concept; ambiguous, patternless sets
(high entropy) give many guesses, whatever the number of examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
STRAY = 0.1


def concept_sets():
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
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")  # (hyps, 100)
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
# Intervals enter only through their size and how many examples they contain, so they
# are pooled into (size, examples-contained) groups; each rule is its own group.
N_GROUPS = N_RULES + 5 * 100
GROUP_SIZE = np.r_[CONCEPT_SIZE[:N_RULES], np.tile(np.arange(1, 101), 5)]
GROUP_LOG_MEMBER = np.log((1 - STRAY) / GROUP_SIZE + STRAY / 100)
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(500)]
BASE_PRIOR = np.r_[np.full(N_RULES, -np.log(N_RULES)), np.full(500, -np.log(N_INTERVALS))]
MAX_ENTROPY = np.log(N_RULES + N_INTERVALS)

with pm.Model() as model:
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_count = pm.Data("group_count", np.zeros((1, N_GROUPS)))  # (sets, groups): concepts in group
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_count = pm.Data("cell_count", np.zeros((1, N_GROUPS)))  # (cells, groups): ... also containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    log_prior = BASE_PRIOR + IS_RULE * pt.log(p_mix) + (1 - IS_RULE) * pt.log1p(-p_mix)
    log_joint = log_prior[None, :] + evidence_strength * group_log_lik  # per concept in group
    log_z = pt.log((pt.exp(log_joint - pt.max(log_joint, axis=1, keepdims=True)) * group_count).sum(axis=1))
    log_post = log_joint - pt.max(log_joint, axis=1, keepdims=True) - log_z[:, None]  # per concept
    post = pt.exp(log_post)
    in_concept = (post[cell_set] * cell_count).sum(axis=1)  # (cells,)
    entropy = -(group_count * post * log_post).sum(axis=1) / MAX_ENTROPY  # (sets,), in [0, 1]

    guess_base = pm.Normal("guess_base", mu=-2.0, sigma=1.5)
    guess_slope = pm.Normal("guess_slope", mu=0.0, sigma=3.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_base + guess_slope * entropy)[cell_set]  # (cells,)
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    interval_size = CONCEPT_SIZE[N_RULES:].astype(int)
    group_of = []
    log_lik = np.zeros((len(unique), N_GROUPS))
    count = np.zeros((len(unique), N_GROUPS))
    for i, s in enumerate(unique):
        n_in = MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype(int)
        g = np.r_[np.arange(N_RULES), N_RULES + 100 * n_in[N_RULES:] + interval_size - 1]
        group_of.append(g)
        g_n_in = np.zeros(N_GROUPS)
        g_n_in[:N_RULES] = n_in[:N_RULES]
        g_n_in[N_RULES:] = np.repeat(np.arange(5), 100)
        log_lik[i] = g_n_in * GROUP_LOG_MEMBER + (len(s) - g_n_in) * np.log(STRAY / 100)
        count[i] = np.bincount(g, minlength=N_GROUPS)
    cell_count = np.stack([
        np.bincount(group_of[index[s]], weights=MEMBERSHIP[:, t - 1], minlength=N_GROUPS)
        for s, t in cells
    ])
    return {
        "group_log_lik": log_lik,
        "group_count": count,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_count": cell_count,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
