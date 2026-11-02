"""Cohesion-selected metric similarity.

People judge the target by its similarity to the examples, but on whichever
single metric of number similarity the examples themselves cluster on most
surprisingly: closeness on the number line (graded, exponentially decaying
gradient) or sharing the same remainder modulo some k from 2 to 10 (parity,
last digit, multiples of 7, ...). Attention goes to the metric on which the
examples are more alike than random numbers would be (a softmax over metrics
of each metric's cohesion surprise), and "yes" is in proportion to the
target's similarity to the nearest example on the attended metric.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPS = 0.1                      # smoothing of pairwise similarity
MODULI = list(range(2, 11))    # residue metrics: mod 2 ... mod 10
NUMBERS = np.arange(1, 101)
ABS_DIFF = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")
MAXN = 4


def _residue_cohesion(ex, k):
    """Sum over later examples of log smoothed mean same-residue rate vs chance."""
    base = (1 - EPS) * np.mean((NUMBERS[:, None] % k) == (NUMBERS[None, :] % k)) + EPS
    tot = 0.0
    for j in range(1, len(ex)):
        s = np.mean([ex[j] % k == ex[i] % k for i in range(j)])
        tot += np.log((1 - EPS) * s + EPS) - np.log(base)
    return tot


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    pair_d = np.zeros((S, MAXN, MAXN))      # |x_j - x_i| for i < j
    pair_m = np.zeros((S, MAXN, MAXN))      # mask of i < j pairs
    later = np.zeros((S, MAXN))             # mask of examples j >= 2
    res_coh = np.zeros((S, len(MODULI)))
    for si, ex in enumerate(unique):
        for j in range(1, len(ex)):
            later[si, j] = 1.0
            for i in range(j):
                pair_d[si, j, i] = abs(ex[j] - ex[i])
                pair_m[si, j, i] = 1.0
        res_coh[si] = [_residue_cohesion(ex, k) for k in MODULI]
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cidx = {c: i for i, c in enumerate(cells)}
    dmin = np.zeros(len(cells))
    res_match = np.zeros((len(cells), len(MODULI)))
    for ci, (ex, t) in enumerate(cells):
        dmin[ci] = min(abs(t - x) for x in ex)
        res_match[ci] = [float(any(t % k == x % k for x in ex)) for k in MODULI]
    return {
        "pair_d": pair_d,
        "pair_m": pair_m,
        "later": later,
        "res_coh": res_coh,
        "cell_set": np.array([sidx[s] for s, _ in cells], dtype="int64"),
        "cell_dmin": dmin,
        "cell_res_match": res_match,
        "row_cell": np.array([cidx[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    pair_d = pm.Data("pair_d", np.zeros((1, MAXN, MAXN)))
    pair_m = pm.Data("pair_m", np.zeros((1, MAXN, MAXN)))
    later = pm.Data("later", np.zeros((1, MAXN)))
    res_coh = pm.Data("res_coh", np.zeros((1, len(MODULI))))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_dmin = pm.Data("cell_dmin", np.zeros(1))
    cell_res_match = pm.Data("cell_res_match", np.zeros((1, len(MODULI))))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Number-line gradient width, attention sharpness, prior pull of the
    # number line over residue metrics, and the yes-rate floor and ceiling.
    log_w = pm.Normal("log_w", np.log(10.0), 1.0)
    w = pt.exp(log_w)
    beta = pm.HalfNormal("beta", 1.0)
    b_line = pm.Normal("b_line", 0.0, 2.0)
    floor = pm.Beta("floor", 1.0, 9.0)
    ceil_frac = pm.Beta("ceil_frac", 2.0, 2.0)
    ceiling = floor + (1 - floor) * ceil_frac

    # Number-line cohesion: later examples' smoothed mean similarity to
    # earlier ones, relative to the similarity of a random pair.
    base_line = (1 - EPS) * pt.mean(pt.exp(-pt.as_tensor(ABS_DIFF) / w)) + EPS
    sim_pairs = pt.exp(-pair_d / w) * pair_m
    n_prev = pt.maximum(pair_m.sum(axis=2), 1.0)
    mean_sim = sim_pairs.sum(axis=2) / n_prev
    line_coh = (later * (pt.log((1 - EPS) * mean_sim + EPS) - pt.log(base_line))).sum(axis=1)

    # Attention over metrics: softmax of cohesion surprise (plus line prior).
    scores = pt.concatenate([(b_line + beta * line_coh)[:, None], beta * res_coh], axis=1)
    attn = pt.special.softmax(scores, axis=1)            # (sets, 1 + K)

    # Target's similarity to its nearest example on each metric.
    sim_t = pt.concatenate([pt.exp(-cell_dmin / w)[:, None], cell_res_match], axis=1)
    sim = (attn[cell_set] * sim_t).sum(axis=1)

    p_cell = floor + (ceiling - floor) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
