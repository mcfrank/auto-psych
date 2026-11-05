"""Place-value context-model exemplar similarity.

People do not infer a rule; they judge a target by its summed similarity to the
stored examples (Medin-Schaffer / Nosofsky context model). Each number is
represented by its tens digit, its units digit, its parity and its magnitude;
similarity to an example shrinks multiplicatively with each mismatching
dimension (inferred attention per dimension). "Yes" grows with total similarity
relative to an internal criterion, so sharing a tens digit with one example
earns partial endorsement even when the examples' common final digit is broken.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # max examples per set


def _tens(n):
    return n // 10


def prepare_observed(rows):
    T = len(rows)
    tens_mis = np.zeros((T, K))
    units_mis = np.zeros((T, K))
    par_mis = np.zeros((T, K))
    mag = np.zeros((T, K))
    mask = np.zeros((T, K))
    y = np.zeros(T, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(float(r["target"]))
        for j, e in enumerate(ex[:K]):
            tens_mis[i, j] = float(_tens(e) != _tens(t))
            units_mis[i, j] = float(e % 10 != t % 10)
            par_mis[i, j] = float(e % 2 != t % 2)
            mag[i, j] = abs(e - t) / 10.0
            mask[i, j] = 1.0
        y[i] = int(float(r["rating"]))
    return {
        "tens_mis": tens_mis,
        "units_mis": units_mis,
        "par_mis": par_mis,
        "mag_dist": mag,
        "ex_mask": mask,
        "rating": y,
    }


with pm.Model() as model:
    tens_mis = pm.Data("tens_mis", np.zeros((1, K), dtype="float64"))
    units_mis = pm.Data("units_mis", np.zeros((1, K), dtype="float64"))
    par_mis = pm.Data("par_mis", np.zeros((1, K), dtype="float64"))
    mag_dist = pm.Data("mag_dist", np.zeros((1, K), dtype="float64"))
    ex_mask = pm.Data("ex_mask", np.ones((1, K), dtype="float64"))

    # attention (mismatch penalty, in log-similarity units) per dimension
    w_tens = pm.HalfNormal("w_tens", sigma=3.0)
    w_units = pm.HalfNormal("w_units", sigma=3.0)
    w_par = pm.HalfNormal("w_par", sigma=3.0)
    w_mag = pm.HalfNormal("w_mag", sigma=2.0)
    # criterion (log scale) against which summed similarity is compared
    log_k = pm.Normal("log_k", mu=-1.0, sigma=2.0)
    # response lapse
    lapse = pm.Beta("lapse", alpha=1.5, beta=15.0)

    log_sim = -(w_tens * tens_mis + w_units * units_mis + w_par * par_mis + w_mag * mag_dist)
    S = pt.sum(pt.exp(log_sim) * ex_mask, axis=1)
    p_core = S / (S + pt.exp(log_k))
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1.0 - lapse) * p_core, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
