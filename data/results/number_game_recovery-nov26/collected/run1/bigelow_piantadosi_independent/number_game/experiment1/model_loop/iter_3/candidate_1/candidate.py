"""Set-adaptive attention exemplar model (GCM-style).

People judge membership by summed similarity to the stored examples, with a
similarity metric that adapts to the set: attention goes to the categorical
dimensions on which the examples agree (final digit, parity, remainder mod 3,
4, 5) and to numerical magnitude in proportion to how tightly the examples
cluster, magnitude being compared on a log (Weber-like) scale. The yes
probability is a Luce choice between the target's summed similarity and a
fixed criterion.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
MODS = [10, 2, 3, 4, 5]  # final digit, parity, mod 3, mod 4, mod 5
N_DIM = len(MODS)
S0 = 0.2  # floor on the examples' log-spread (fixed)


def _set_stats(examples):
    agree = np.zeros(N_DIM)
    for k, m in enumerate(MODS):
        vals = [x % m for x in examples]
        modal = max(vals.count(v) for v in set(vals)) / len(vals)
        agree[k] = max(0.0, (modal - 1.0 / m) / (1.0 - 1.0 / m))
    logs = np.log(examples)
    spread = float(logs.max() - logs.min())
    return agree, 1.0 / (spread + S0)


def prepare_observed(rows):
    n = len(rows)
    mis = np.zeros((n, MAX_EX, N_DIM))
    logd = np.zeros((n, MAX_EX))
    mask = np.zeros((n, MAX_EX))
    agree = np.zeros((n, N_DIM))
    mag_att = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        key = tuple(ex)
        if key not in cache:
            cache[key] = _set_stats(np.array(ex, dtype=float))
        agree[i], mag_att[i] = cache[key]
        for j, x in enumerate(ex[:MAX_EX]):
            mask[i, j] = 1.0
            logd[i, j] = abs(np.log(t) - np.log(x))
            for k, m in enumerate(MODS):
                mis[i, j, k] = 0.0 if t % m == x % m else 1.0
        rating[i] = int(float(r["rating"]))
    return {"cat_mismatch": mis, "log_dist": logd, "example_mask": mask,
            "dim_agreement": agree, "mag_attention": mag_att, "rating": rating}


with pm.Model() as model:
    mis = pm.Data("cat_mismatch", np.zeros((1, MAX_EX, N_DIM), dtype="float64"))
    logd = pm.Data("log_dist", np.zeros((1, MAX_EX), dtype="float64"))
    mask = pm.Data("example_mask", np.ones((1, MAX_EX), dtype="float64"))
    agree = pm.Data("dim_agreement", np.ones((1, N_DIM), dtype="float64"))
    mag_att = pm.Data("mag_attention", np.ones(1, dtype="float64"))

    # Categorical sensitivity, attention concentration on agreeing dimensions,
    # magnitude sensitivity, Luce criterion and response determinism.
    lam = pm.HalfNormal("lam", sigma=3.0)
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=0.5)
    mu = pm.HalfNormal("mu", sigma=3.0)
    log_kappa = pm.Normal("log_kappa", mu=-2.0, sigma=2.0)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)

    att = lam * pt.exp(gamma * pt.log(agree + 1e-3))            # (n, dim)
    dist = (mis * att[:, None, :]).sum(axis=2) \
        + mu * mag_att[:, None] * logd                           # (n, ex)
    sim = (pt.exp(-dist) * mask).sum(axis=1) / mask.sum(axis=1)
    log_s = pt.log(sim + 1e-12)
    p = pm.math.sigmoid(beta * (log_s - log_kappa))
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
