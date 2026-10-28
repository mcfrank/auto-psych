"""Running-prototype delta rule.

People keep a single running summary (prototype) of the program's numbers --
their size, parity, final digit and small factors -- and revise it as each
example arrives, moving it only part of the way (one learning rate) toward each
new example.  A target is endorsed by how well it fits the final summary, so
order matters: a slow learner stays anchored on the first example (primacy), a
fast one is dominated by the last (recency), and properties shared by every
example survive at full strength while one-off properties are diluted.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4
FACTORS = list(range(3, 11))


def _features(x):
    """Binary property features of a number (parity, final digit, factors 3-10)."""
    f = [1.0 if x % 2 == 0 else 0.0, 1.0 if x % 2 == 1 else 0.0]
    f += [1.0 if x % 10 == d else 0.0 for d in range(10)]
    f += [1.0 if x % g == 0 else 0.0 for g in FACTORS]
    return f


NPAR, NDIG, NFAC = 2, 10, len(FACTORS)
NF = NPAR + NDIG + NFAC


def prepare_observed(rows):
    n = len(rows)
    ex_f = np.zeros((n, MAXN, NF))
    ex_m = np.zeros((n, MAXN))
    expo = np.zeros((n, MAXN))
    first = np.zeros((n, MAXN))
    valid = np.zeros((n, MAXN))
    t_f = np.zeros((n, NF))
    t_m = np.zeros(n)
    y = np.zeros(n, dtype="int64")
    for i, r in enumerate(rows):
        xs = [int(v) for v in str(r["number_set"]).split(",")][:MAXN]
        k = len(xs)
        for j, x in enumerate(xs):
            ex_f[i, j] = _features(x)
            ex_m[i, j] = x / 100.0
            expo[i, j] = k - 1 - j
            first[i, j] = 1.0 if j == 0 else 0.0
            valid[i, j] = 1.0
        t = int(r["target"])
        t_f[i] = _features(t)
        t_m[i] = t / 100.0
        y[i] = int(float(r["rating"]))
    return {"ex_feat": ex_f, "ex_mag": ex_m, "ex_expo": expo, "ex_first": first,
            "ex_valid": valid, "tg_feat": t_f, "tg_mag": t_m, "rating": y}


with pm.Model() as model:
    ex_feat = pm.Data("ex_feat", np.zeros((1, MAXN, NF)))
    ex_mag = pm.Data("ex_mag", np.zeros((1, MAXN)))
    ex_expo = pm.Data("ex_expo", np.zeros((1, MAXN)))
    ex_first = pm.Data("ex_first", np.ones((1, MAXN)))
    ex_valid = pm.Data("ex_valid", np.ones((1, MAXN)))
    tg_feat = pm.Data("tg_feat", np.zeros((1, NF)))
    tg_mag = pm.Data("tg_mag", np.zeros(1))

    # Learning rate of the delta rule: < 0.5 anchors on the first example,
    # > 0.5 lets the last example dominate.
    alpha = pm.Beta("alpha", alpha=2.0, beta=2.0)
    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    w_par = pm.HalfNormal("w_par", sigma=3.0)
    w_dig = pm.HalfNormal("w_dig", sigma=3.0)
    w_fac = pm.HalfNormal("w_fac", sigma=3.0)

    # Weight each example carries in the final prototype after sequential updates:
    # first example (1-a)^(n-1), example k>1  a(1-a)^(n-k).
    decay = (1.0 - alpha) ** ex_expo
    w = pt.switch(ex_first > 0.5, decay, alpha * decay) * ex_valid  # (N, MAXN)

    proto_f = pt.sum(w[:, :, None] * ex_feat, axis=1)  # (N, NF)
    proto_m = pt.sum(w * ex_mag, axis=1)

    # Mismatch: prototype properties the target lacks; distance in size.
    lack = proto_f * (1.0 - tg_feat)
    d_par = pt.sum(lack[:, :NPAR], axis=1)
    d_dig = pt.sum(lack[:, NPAR:NPAR + NDIG], axis=1)
    d_fac = pt.sum(lack[:, NPAR + NDIG:], axis=1)
    d_mag = pt.sqrt((proto_m - tg_mag) ** 2 + 1e-4)

    logit = intercept - w_mag * d_mag - w_par * d_par - w_dig * d_dig - w_fac * d_fac
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
