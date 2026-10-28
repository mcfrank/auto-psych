"""Refinement of concept_learning_with_guessing: Bayesian concept learning over
42 weighted rules and all 5,050 intervals of 1..100 (size principle) with a
guessing lapse, where the believed share of examples that are random noise
(drawn uniformly from 1..100 rather than from the concept) is inferred instead
of fixed at 10%. A higher noise belief lets a rule survive an example that
violates it and weakens the size principle's preference for tight concepts.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZES = MEMBERSHIP[:N_RULES].sum(axis=1)
# An interval's likelihood depends only on its width and how many examples it
# holds, so intervals are grouped into (width, examples-inside) cells; each set
# gets the count of intervals per cell, and per target the count containing it.
MAX_EX = 4
CELL_WIDTH = np.repeat(np.arange(1, 101), MAX_EX + 1).astype("float64")
CELL_K = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
N_CELLS = CELL_WIDTH.size

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside each rule
    cell_log_count = pm.Data("cell_log_count", np.zeros((1, N_CELLS)))  # (sets, cells): log #intervals
    pair_frac = pm.Data("pair_frac", np.zeros((1, N_CELLS)))  # (set-target pairs, cells): share containing target
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))  # set index of each pair
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))  # pair index of each response
    n_ex = pm.Data("n_ex", np.ones(1))  # (sets,): number of examples
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    epsilon = pm.Beta("epsilon", 2, 12)  # believed share of examples that are random

    # Each in-concept example has likelihood (1-eps)/|h| + eps/100; each outside one eps/100.
    log_out = pt.log(epsilon / 100.0)
    rule_log_in = pt.log((1 - epsilon) / RULE_SIZES + epsilon / 100.0)
    log_rules = (pt.log(p_mix) + pt.log(rule_weights)[None, :]
                 + rule_n_in * rule_log_in[None, :] + (n_ex[:, None] - rule_n_in) * log_out)
    cell_log_in = pt.log((1 - epsilon) / CELL_WIDTH + epsilon / 100.0)
    log_cells = (pt.log1p(-p_mix) - np.log(N_INTERVALS) + cell_log_count
                 + CELL_K[None, :] * cell_log_in[None, :] + (n_ex[:, None] - CELL_K[None, :]) * log_out)
    log_joint = pt.concatenate([log_rules, log_cells], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Only the observed (set, target) pairs are evaluated.
    rule_in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[pair_cell]
    interval_in_concept = (pair_frac * posterior[pair_set, N_RULES:]).sum(axis=1)
    concept_p = pt.clip((rule_in_concept + interval_in_concept)[row_pair], 0.0, 1.0)

    guess_rate = pm.Beta("guess_rate", 2, 8)
    guess_bias = pm.Beta("guess_bias", 2, 2)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - guess_rate) * concept_p + guess_rate * guess_bias, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_mem = MEMBERSHIP[N_RULES:]
    widths = interval_mem.sum(axis=1).astype(int)
    log_count, target_frac = [], []
    for s in unique:
        k = interval_mem[:, np.array(s) - 1].sum(axis=1).astype(int)
        cell = (widths - 1) * (MAX_EX + 1) + k
        counts = np.bincount(cell, minlength=N_CELLS).astype("float64")
        containing = np.zeros((100, N_CELLS))
        np.add.at(containing.T, cell, interval_mem)
        log_count.append(np.where(counts > 0, np.log(np.maximum(counts, 1.0)), -1e4))
        target_frac.append(containing / np.maximum(counts, 1.0)[None, :])
    row_cells = [index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)]
    pairs = list(dict.fromkeys(row_cells))
    pair_index = {c: i for i, c in enumerate(pairs)}
    return {
        "rule_n_in": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "cell_log_count": np.stack(log_count),
        "pair_frac": np.stack([target_frac[c // 100][c % 100] for c in pairs]),
        "pair_set": np.array([c // 100 for c in pairs], dtype="int64"),
        "pair_cell": np.array(pairs, dtype="int64"),
        "row_pair": np.array([pair_index[c] for c in row_cells], dtype="int64"),
        "n_ex": np.array([float(len(s)) for s in unique]),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
