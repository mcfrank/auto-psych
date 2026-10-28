"""Feature-distribution prototype (category density) account.

People summarise the example set as one statistical category prototype: the
examples' central size and spread, and how often each property (final digit,
parity, divisibility by 3-10) occurs among the examples, smoothed toward how
common that property is among numbers in general. They say "yes" in
proportion to how typical the target is of that prototype relative to an
arbitrary number, so properties shared by every example become sharply (and
increasingly, with set size) diagnostic, while size generalisation broadens
with the examples' spread.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

FACTORS = list(range(3, 11))


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = len(ex)
    arr = np.array(ex, dtype=float) / 100.0
    feats = {
        "n_ex": float(n),
        "mag_sq": float((t / 100.0 - arr.mean()) ** 2),
        "mag_var": float(arr.var()),
        "c_digit": float(sum(1 for x in ex if x % 10 == t % 10)),
        "c_parity": float(sum(1 for x in ex if x % 2 == t % 2)),
    }
    for k in FACTORS:
        tdiv = t % k == 0
        feats[f"c_f{k}"] = float(sum(1 for x in ex if (x % k == 0) == tdiv))
        feats[f"base_f{k}"] = (1.0 / k) if tdiv else (1.0 - 1.0 / k)
    return feats


with pm.Model() as model:
    n_ex = pm.Data("n_ex", np.zeros(1, dtype="float64"))
    mag_sq = pm.Data("mag_sq", np.zeros(1, dtype="float64"))
    mag_var = pm.Data("mag_var", np.zeros(1, dtype="float64"))
    c_digit = pm.Data("c_digit", np.zeros(1, dtype="float64"))
    c_parity = pm.Data("c_parity", np.zeros(1, dtype="float64"))
    c_f = [pm.Data(f"c_f{k}", np.zeros(1, dtype="float64")) for k in FACTORS]
    base_f = [pm.Data(f"base_f{k}", np.ones(1, dtype="float64")) for k in FACTORS]

    # Prototype parameters: smoothing toward base rates, size-spread floor.
    conc = pm.LogNormal("conc", mu=0.0, sigma=1.0)
    s0 = pm.HalfNormal("s0", sigma=0.3)
    # Decision: intercept and how much each prototype dimension's typicality
    # contributes (one weight per dimension).
    b0 = pm.Normal("b0", mu=0.0, sigma=2.0)
    w_mag = pm.HalfNormal("w_mag", sigma=1.0)
    w_dig = pm.HalfNormal("w_dig", sigma=1.0)
    w_par = pm.HalfNormal("w_par", sigma=1.0)
    w_fac = pm.HalfNormal("w_fac", sigma=1.0)

    denom = n_ex + conc
    # Size: Gaussian prototype vs uniform over 1-100 (width 1 in /100 units).
    v = mag_var + s0 ** 2
    l_mag = -0.5 * mag_sq / v - 0.5 * pt.log(2 * np.pi * v)
    l_dig = pt.log((c_digit + conc * 0.1) / denom) - np.log(0.1)
    l_par = pt.log((c_parity + conc * 0.5) / denom) - np.log(0.5)
    l_fac = sum(pt.log((c + conc * b) / denom) - pt.log(b) for c, b in zip(c_f, base_f))

    score = b0 + w_mag * l_mag + w_dig * l_dig + w_par * l_par + w_fac * l_fac
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(score), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
