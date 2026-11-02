"""Additive-clustering exemplar similarity.

People do not infer a hidden concept; they compare the target to each stored
example and judge similarity by the stored number categories the two numbers
share (rules such as parity, multiples, squares, powers, last digit, digits
contained, and intervals of the number line), each shared category weighted by
the inverse of its size. They say yes the more the target resembles every
example (geometric mean of pairwise similarities).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    r = [NUMS % 2 == 1, NUMS % 2 == 0]
    r.append(np.isin(NUMS, [k * k for k in range(1, 11)]))
    r.append(np.isin(NUMS, [k ** 3 for k in range(1, 5)]))
    r.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS]))
    for b in range(2, 11):
        r.append(np.isin(NUMS, [b ** k for k in range(1, 8) if b ** k <= N]))
    p2 = [2 ** k for k in range(1, 7)]
    r.append(np.isin(NUMS, [x for x in p2 if x != 32]))
    r.append(np.isin(NUMS, p2 + [37]))
    for m in list(range(3, 10)) + [11, 12]:
        r.append(NUMS % m == 0)
    for d in range(10):
        r.append(NUMS % 10 == d)
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
    return np.array(r, dtype=float)


def _similarity_tables():
    R = _rules()
    w = 1.0 / np.maximum(R.sum(1), 1.0)
    rule_sim = (R * w[:, None]).T @ R  # sum of 1/|c| over shared rules
    # Intervals [a,b] containing both i and j, weight 1/(b-a+1).
    lo = np.minimum.outer(NUMS, NUMS)
    hi = np.maximum.outer(NUMS, NUMS)
    widths = np.arange(1, N + 1)
    interval_sim = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            a_max, b_min = lo[i, j], hi[i, j]
            s = 0.0
            for a in range(1, a_max + 1):
                b = np.arange(b_min, N + 1)
                s += np.sum(1.0 / (b - a + 1))
            interval_sim[i, j] = interval_sim[j, i] = s
    return rule_sim / rule_sim.max(), interval_sim / interval_sim.max()


RULE_SIM, INT_SIM = _similarity_tables()
MAXE = 4


def prepare_observed(rows):
    n = len(rows)
    rs = np.zeros((n, MAXE))
    isim = np.zeros((n, MAXE))
    mask = np.zeros((n, MAXE))
    y = np.zeros(n, dtype="int64")
    for k, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")]
        t = int(row["target"])
        for j, e in enumerate(ex[:MAXE]):
            rs[k, j] = RULE_SIM[t - 1, e - 1]
            isim[k, j] = INT_SIM[t - 1, e - 1]
            mask[k, j] = 1.0
        y[k] = int(float(row["rating"]))
    return {"pair_rule_sim": rs, "pair_interval_sim": isim, "example_mask": mask, "rating": y}


with pm.Model() as model:
    pair_rule_sim = pm.Data("pair_rule_sim", np.zeros((1, MAXE)))
    pair_interval_sim = pm.Data("pair_interval_sim", np.zeros((1, MAXE)))
    example_mask = pm.Data("example_mask", np.ones((1, MAXE)))

    # Relative weight of shared number-line stretches vs shared rule categories.
    log_gamma = pm.Normal("log_gamma", mu=0.0, sigma=1.5)
    # Response mapping from log similarity to yes.
    intercept = pm.Normal("intercept", mu=0.0, sigma=3.0)
    slope = pm.HalfNormal("slope", sigma=3.0)

    sim = pair_rule_sim + pt.exp(log_gamma) * pair_interval_sim
    log_sim = pt.log(sim + 1e-4) * example_mask
    mean_log_sim = log_sim.sum(axis=1) / example_mask.sum(axis=1)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + slope * mean_log_sim), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
