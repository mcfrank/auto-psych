"""Consistency-driven selective-attention similarity.

People do not infer a hidden rule; they judge a target by its similarity to the
examples, comparing numbers along a few dimensions (magnitude on the number
line, parity, final digit, shared written digits, sharing a common factor), and
they focus attention on whichever dimensions the examples agree on with one
another. A target is endorsed to the extent that it matches the examples on the
dimensions along which the examples are mutually similar; magnitude similarity
falls off exponentially with numerical distance, and the sharpness of the
attention shift is inferred.
"""
import math
import itertools
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_DISCRETE = 4  # parity, final digit, shared digit, common factor


def _discrete_match(a, b):
    return np.array([
        float(a % 2 == b % 2),
        float(a % 10 == b % 10),
        float(len(set(str(a)) & set(str(b))) > 0),
        float(math.gcd(a, b) >= 3),
    ])


def prepare_observed(rows):
    n = len(rows)
    tgt_sim = np.zeros((n, N_DISCRETE))
    cons = np.zeros((n, N_DISCRETE))
    dist = np.zeros((n, 4))
    dmask = np.zeros((n, 4))
    pdist = np.zeros((n, 6))
    pmask = np.zeros((n, 6))
    single = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    for i, r in enumerate(rows):
        xs = [int(v) for v in str(r["number_set"]).split(",")]
        t = int(r["target"])
        tgt_sim[i] = np.mean([_discrete_match(t, x) for x in xs], axis=0)
        for j, x in enumerate(xs[:4]):
            dist[i, j] = abs(t - x) / 10.0
            dmask[i, j] = 1.0
        pairs = list(itertools.combinations(xs[:4], 2))
        if pairs:
            cons[i] = np.mean([_discrete_match(a, b) for a, b in pairs], axis=0)
            for k, (a, b) in enumerate(pairs):
                pdist[i, k] = abs(a - b) / 10.0
                pmask[i, k] = 1.0
        else:
            cons[i] = 1.0
            single[i] = 1.0
        rating[i] = int(float(r["rating"]))
    return {
        "tgt_sim": tgt_sim, "cons": cons, "dist": dist, "dmask": dmask,
        "pdist": pdist, "pmask": pmask, "single": single, "rating": rating,
    }


with pm.Model() as model:
    tgt_sim = pm.Data("tgt_sim", np.zeros((1, N_DISCRETE)))
    cons = pm.Data("cons", np.zeros((1, N_DISCRETE)))
    dist = pm.Data("dist", np.zeros((1, 4)))
    dmask = pm.Data("dmask", np.ones((1, 4)))
    pdist = pm.Data("pdist", np.zeros((1, 6)))
    pmask = pm.Data("pmask", np.zeros((1, 6)))
    single = pm.Data("single", np.ones(1))

    # magnitude generalisation gradient width (in units of 10)
    width = pm.LogNormal("width", mu=0.0, sigma=1.0)
    # how strongly attention follows within-set consistency
    focus = pm.HalfNormal("focus", sigma=5.0)
    # baseline salience of each dimension (magnitude fixed at 0)
    salience = pm.Normal("salience", mu=0.0, sigma=1.5, shape=N_DISCRETE)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    mag_sim = pt.sum(pt.exp(-dist / width) * dmask, axis=1) / pt.sum(dmask, axis=1)
    pair_sim = pt.sum(pt.exp(-pdist / width) * pmask, axis=1) / pt.maximum(pt.sum(pmask, axis=1), 1.0)
    mag_cons = single + (1.0 - single) * pair_sim

    sims = pt.concatenate([mag_sim[:, None], tgt_sim], axis=1)
    consis = pt.concatenate([mag_cons[:, None], cons], axis=1)
    sal = pt.concatenate([pt.zeros(1), salience])
    att = pt.special.softmax(sal[None, :] + focus * consis, axis=1)
    score = pt.sum(att * sims, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * score), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
