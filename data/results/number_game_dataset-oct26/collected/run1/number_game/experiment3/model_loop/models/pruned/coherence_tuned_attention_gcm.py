"""Coherence-tuned attention in an exemplar (GCM-style) similarity judgment.

People do not infer a rule; they judge the target by its average similarity to
the stored examples, comparing it with each example on three separately coded
aspects: final digit (same digit most alike, same parity partly), decade
(similarity falling off with decades apart), and whether the two share a
divisor of 3 or more. Attention across these aspects is tuned by the stored
examples themselves: people attend most to whichever aspect the examples cohere
on (pairwise shared digits or parity, nearby decades, shared divisors), so a
match on that aspect counts heavily and one on an aspect the examples vary on
counts little; a lone example leaves attention at its default split.
"""
import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    f = {}
    for k in range(K):
        on = k < len(xs)
        x = xs[k] if on else 1
        f[f"ex_mask_{k}"] = 1.0 if on else 0.0
        f[f"ex_same_dig_{k}"] = 1.0 if on and x % 10 == t % 10 else 0.0
        f[f"ex_same_par_{k}"] = 1.0 if on and x % 2 == t % 2 and x % 10 != t % 10 else 0.0
        f[f"ex_dec_dist_{k}"] = float(abs(x // 10 - t // 10)) if on else 0.0
        f[f"ex_share_div_{k}"] = 1.0 if on and math.gcd(x, t) >= 3 else 0.0
    pairs = [(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    m = float(max(len(pairs), 1))
    f["pair_same_dig"] = sum(a % 10 == b % 10 for a, b in pairs) / m
    f["pair_same_par"] = sum(a % 2 == b % 2 and a % 10 != b % 10 for a, b in pairs) / m
    f["pair_dec_dist"] = sum(abs(a // 10 - b // 10) for a, b in pairs) / m
    f["pair_share_div"] = sum(math.gcd(a, b) >= 3 for a, b in pairs) / m
    f["multi_example"] = 1.0 if pairs else 0.0
    return f


with pm.Model() as model:
    def slots(name):
        return pt.stack([pm.Data(f"{name}_{k}", np.zeros(1)) for k in range(K)], axis=1)

    mask = slots("ex_mask")
    sdig = slots("ex_same_dig")
    spar = slots("ex_same_par")
    ddist = slots("ex_dec_dist")
    sdiv = slots("ex_share_div")
    p_dig = pm.Data("pair_same_dig", np.zeros(1))
    p_par = pm.Data("pair_same_par", np.zeros(1))
    p_dec = pm.Data("pair_dec_dist", np.zeros(1))
    p_div = pm.Data("pair_share_div", np.zeros(1))
    multi = pm.Data("multi_example", np.zeros(1))

    s_par = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    width = pm.LogNormal("decade_width", mu=0.0, sigma=0.75)

    n_ex = pt.sum(mask, axis=1)
    sim_dig = pt.sum((sdig + s_par * spar) * mask, axis=1) / n_ex
    sim_mag = pt.sum(pt.exp(-ddist / width) * mask, axis=1) / n_ex
    sim_div = pt.sum(sdiv * mask, axis=1) / n_ex

    coh = pt.stack([p_dig + s_par * p_par, pt.exp(-p_dec / width), p_div], axis=1)
    base = pm.Normal("att_base", mu=0.0, sigma=1.5, shape=2)
    tau = pm.HalfNormal("att_tuning", sigma=3.0)
    logits = pt.concatenate([base, pt.zeros(1)])[None, :] + tau * multi[:, None] * coh
    att = pt.special.softmax(logits, axis=1)
    S = att[:, 0] * sim_dig + att[:, 1] * sim_mag + att[:, 2] * sim_div

    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * S))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)

