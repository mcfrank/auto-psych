"""Span-relative power-law similarity.

People judge a target purely by its closeness on the number line to the span
the examples cover: a target inside the span is fully similar; outside it,
similarity falls as a power law of the distance beyond the nearest edge
measured relative to the span's width (plus an inferred baseline width, so a
single example still generalizes to its neighbours). The falloff steepens with
each additional example. Apart from occasional blind guesses, "yes" follows
this relative-distance similarity.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",") if x.strip()]
    t = int(target)
    lo, hi = min(xs), max(xs)
    d_out = float(max(lo - t, t - hi, 0))
    return {
        "dist_outside": d_out / 100.0,
        "span": (hi - lo) / 100.0,
        "n_examples": float(len(xs)),
    }


with pm.Model() as model:
    dist_outside = pm.Data("dist_outside", np.zeros(1, dtype="float64"))
    span = pm.Data("span", np.zeros(1, dtype="float64"))
    n_examples = pm.Data("n_examples", np.ones(1, dtype="float64"))

    # Baseline width added to the examples' span (on the 0-1 scale).
    base_width = pm.HalfNormal("base_width", sigma=0.2)
    # How steeply similarity falls per example shown.
    steepness = pm.HalfNormal("steepness", sigma=2.0)
    # Blind-guess floor (yes) and ceiling (no).
    guess_yes = pm.Beta("guess_yes", alpha=1.0, beta=5.0)
    guess_no = pm.Beta("guess_no", alpha=1.0, beta=5.0)

    rel = dist_outside / (span + base_width + 1e-3)
    sim = pt.exp(-steepness * n_examples * pt.log1p(rel))
    p = guess_yes + (1.0 - guess_yes - guess_no) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
