"""Digit-consistency attention exemplar model (refines exemplar_magnitude_digit_similarity).

People judge a target by its summed similarity to the remembered examples on
magnitude and final digit, and say "yes" by a Luce ratio S / (S + k). The one
change: attention to the final digit is not fixed but grows with how
consistently the examples share final digits (the fraction of example pairs
with the same last digit), so a last-digit mismatch is penalised far more after
sets like 78,48,88 than after sets whose last digits vary.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",") if x.strip() != ""][:MAXN]
    t = int(target)
    pairs = list(itertools.combinations(ex, 2))
    cons = float(np.mean([a % 10 == b % 10 for a, b in pairs])) if pairs else 0.0
    out = {"digit_consistency": cons}
    for i in range(MAXN):
        if i < len(ex):
            out[f"mag_dist_{i}"] = abs(t - ex[i]) / 10.0
            out[f"digit_mis_{i}"] = 0.0 if (t % 10) == (ex[i] % 10) else 1.0
            out[f"ex_mask_{i}"] = 1.0
        else:
            out[f"mag_dist_{i}"] = 0.0
            out[f"digit_mis_{i}"] = 0.0
            out[f"ex_mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    mag = pt.stack([pm.Data(f"mag_dist_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)
    mis = pt.stack([pm.Data(f"digit_mis_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)
    mask = pt.stack([pm.Data(f"ex_mask_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)
    cons = pm.Data("digit_consistency", np.zeros(1, dtype="float64"))

    # Magnitude gradient (per 10 units), baseline digit-mismatch penalty, the
    # extra penalty when examples fully agree on final digit, Luce criterion.
    c_mag = pm.HalfNormal("c_mag", sigma=1.0)
    c_digit = pm.HalfNormal("c_digit", sigma=3.0)
    c_cons = pm.HalfNormal("c_cons", sigma=3.0)
    log_k = pm.Normal("log_k", mu=0.0, sigma=2.0)

    digit_pen = (c_digit + c_cons * cons)[:, None]
    log_sim = pt.switch(mask > 0.5, -c_mag * mag - digit_pen * mis, -1e3)
    log_S = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(log_S - log_k), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
