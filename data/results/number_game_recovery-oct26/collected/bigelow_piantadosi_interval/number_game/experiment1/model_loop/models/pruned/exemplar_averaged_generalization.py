"""Exemplar-averaged generalization.

People store each example as a separate exemplar rather than inferring one
concept from the whole set. Each single example generalizes to the target
through the familiar categories (42 weighted rules and all intervals of 1-100)
that contain both, with smaller categories counting more (size principle for
one example, tempered by an inferred evidence weight, 10% noisy examples).
These single-example generalization strengths are averaged across the examples
(GCM-style summed similarity), and the average belief is mapped to yes/no
through an inferred decision gain on its log-odds. Adding consistent examples
therefore never sharpens generalization as joint Bayesian inference does.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


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
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
# log P(x | rule) for a single example x: (rules, 100)
RULE_LOG_LIK = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
LOG_N_INTERVALS = np.log(len(INTERVALS))

# Intervals are equally likely a priori, so given one example x an interval's
# posterior depends only on its width and whether it contains x. Precompute,
# per width w: BOTH[w, x, t] = # width-w intervals containing x and t,
# ONLY_T[w, x, t] = # containing t but not x, and their row totals for x.
_lo = np.arange(1, 101)[:, None]
_hi = np.arange(1, 101)[None, :]
BOTH = np.zeros((100, 100, 100))
ONLY_T = np.zeros((100, 100, 100))
for _w in range(1, 101):
    _starts = np.arange(1, 102 - _w)
    _mem = ((NUMBERS[None, :] >= _starts[:, None]) & (NUMBERS[None, :] < _starts[:, None] + _w)).astype("float64")
    BOTH[_w - 1] = _mem.T @ _mem
    ONLY_T[_w - 1] = (1 - _mem).T @ _mem
N_WITH_X = BOTH.sum(axis=2) / np.arange(1, 101)[:, None]  # (w, x) intervals containing x
N_WITHOUT_X = np.arange(100, 0, -1)[:, None] - N_WITH_X  # (w, x)
LOG_LIK_IN = np.log((1 - EPSILON) / np.arange(1, 101) + EPSILON / 100)  # (w,)
LOG_LIK_OUT = np.log(EPSILON / 100)


def prepare_observed(rows):
    """Per trial: padded example indices (0-99), a mask, and the target index."""
    n = len(rows)
    ex = np.zeros((n, MAX_SET), dtype="int64")
    mask = np.zeros((n, MAX_SET), dtype="float64")
    for i, r in enumerate(rows):
        xs = [int(v) for v in str(r["number_set"]).split(",")][:MAX_SET]
        ex[i, : len(xs)] = np.array(xs) - 1
        mask[i, : len(xs)] = 1.0
    return {
        "example_idx": ex,
        "example_mask": mask,
        "target_idx": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_SET), dtype="int64"))
    example_mask = pm.Data("example_mask", np.ones((1, MAX_SET), dtype="float64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)

    # Posterior over categories given one stored example x, rules part: (rules, x)
    log_rule = pt.log(p_mix) + pt.log(rule_weights)[:, None] + evidence_weight * RULE_LOG_LIK
    # Intervals: per-width weights for "contains x" and "does not contain x"
    log_iv = pt.log1p(-p_mix) - LOG_N_INTERVALS
    log_in = log_iv + evidence_weight * LOG_LIK_IN  # (w,)
    log_out = log_iv + evidence_weight * LOG_LIK_OUT  # scalar
    shift = pt.max(log_rule)
    shift = pt.maximum(shift, pt.max(log_in))
    w_rule = pt.exp(log_rule - shift)
    w_in = pt.exp(log_in - shift)
    w_out = pt.exp(log_out - shift)
    norm = w_rule.sum(axis=0) + w_in @ N_WITH_X + w_out * N_WITHOUT_X.sum(axis=0)  # (x,)
    num = (
        w_rule.T @ RULE_MEMBERSHIP
        + pt.tensordot(w_in, BOTH, axes=[[0], [0]])
        + w_out * ONLY_T.sum(axis=0)
    )  # (x, t)
    # Single-exemplar generalization G[x, t] = P(t in h | x)
    gen = num / norm[:, None]

    per_example = gen[example_idx, target_idx[:, None]]  # (trials, MAX_SET)
    avg = (per_example * example_mask).sum(axis=1) / example_mask.sum(axis=1)
    belief = pt.clip(avg, 1e-6, 1 - 1e-6)
    belief_logit = pt.log(belief) - pt.log1p(-belief)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)
