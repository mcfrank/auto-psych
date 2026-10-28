"""Conjunctive category-overlap similarity.

People judge a target by its similarity to each example separately, where two
numbers are similar to the degree that they fall together in the same familiar
number categories (42 weighted rules and all ranges of 1..100), with small,
specific shared categories counting more than broad ones by an inferred
similarity gradient (shared category weight ~ size^-alpha). The per-example
similarity G(x, y) = sum_h w_h |h|^-a [x,y in h] / sum_h w_h |h|^-a [x in h]
(1 when y = x). Per-example similarities combine conjunctively (product over
the examples), and p_yes = sigmoid(gain * logit(S) + bias).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_SET = 4


def rule_sets():
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
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, r) for r in RULES], dtype="float64")  # (rules, 100)
LOG_RULE_SIZE = np.log(RULE_MEM.sum(axis=1))
N_INTERVALS = 5050.0

# Number of ranges of each width w containing both x and y: (100*100, widths).
_lo = np.minimum.outer(NUMBERS, NUMBERS).reshape(-1, 1)
_hi = np.maximum.outer(NUMBERS, NUMBERS).reshape(-1, 1)
_w = NUMBERS.reshape(1, -1)
PAIR_RANGE_COUNT = np.maximum(0, np.minimum(_lo, 101 - _w) - np.maximum(1, _hi - _w + 1) + 1).astype("float64")
LOG_WIDTH = np.log(NUMBERS.astype("float64"))

with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_SET), dtype="int64"))  # (trials, 4) example - 1
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_SET)))  # 1 for real examples
    tgt_idx = pm.Data("tgt_idx", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # Similarity gradient: how much more a small shared category counts.
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    rule_w = p_mix * rule_weights * pt.exp(-alpha * LOG_RULE_SIZE)  # (rules,)
    range_w = (1 - p_mix) / N_INTERVALS * pt.exp(-alpha * LOG_WIDTH)  # (widths,)
    rule_num = (pt.as_tensor(RULE_MEM.T) * rule_w[None, :]) @ pt.as_tensor(RULE_MEM)  # (100, 100)
    range_num = (pt.as_tensor(PAIR_RANGE_COUNT) @ range_w).reshape((100, 100))
    num = rule_num + range_num
    denom = pt.diagonal(num)  # all categories containing x
    gen = pt.clip(num / denom[:, None], 1e-9, 1.0)  # G(x, y)
    log_gen = pt.log(gen)

    log_s = (log_gen[ex_idx, tgt_idx[:, None]] * ex_mask).sum(axis=1)
    s = pt.clip(pt.exp(log_s), 1e-6, 1 - 1e-6)
    belief_logit = pt.log(s) - pt.log1p(-s)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    n = len(rows)
    ex_idx = np.zeros((n, MAX_SET), dtype="int64")
    ex_mask = np.zeros((n, MAX_SET))
    for i, r in enumerate(rows):
        ex = [int(x) - 1 for x in str(r["number_set"]).split(",")]
        ex_idx[i, : len(ex)] = ex
        ex_mask[i, : len(ex)] = 1.0
    return {
        "ex_idx": ex_idx,
        "ex_mask": ex_mask,
        "tgt_idx": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
