"""Arithmetic-relation exemplar similarity.

People do not infer a hidden rule; they judge a target by its similarity to the
stored examples, as in an exemplar (context) model, where two numbers are
similar to the extent that they stand in familiar arithmetic relations -- one is
a multiple of the other, they share a common factor of 3 or more, they end in
the same digit, they have the same parity -- and each relation the pair lacks
shrinks their similarity by an inferred factor. The target's average similarity
to the examples, against an internal criterion, sets the chance of "yes", so a
lone example is generalized to its own multiples (6 -> 18, 25 -> 100,
11 -> 55), while numbers sharing none of these relations with any example are
rejected. A small lapse rate covers answers given without consulting memory.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # maximum number of examples in a set


def prepare_observed(rows):
    T = len(rows)
    ratio_mis = np.zeros((T, K))
    factor_mis = np.zeros((T, K))
    digit_mis = np.zeros((T, K))
    parity_mis = np.zeros((T, K))
    ex_mask = np.zeros((T, K))
    y = np.zeros(T, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(float(r["target"]))
        for j, e in enumerate(ex[:K]):
            ratio_mis[i, j] = 0.0 if (t % e == 0 or e % t == 0) else 1.0
            factor_mis[i, j] = 0.0 if math.gcd(t, e) >= 3 else 1.0
            digit_mis[i, j] = 0.0 if t % 10 == e % 10 else 1.0
            parity_mis[i, j] = 0.0 if t % 2 == e % 2 else 1.0
            ex_mask[i, j] = 1.0
        y[i] = int(float(r["rating"]))
    return {
        "ratio_mis": ratio_mis,
        "factor_mis": factor_mis,
        "digit_mis": digit_mis,
        "parity_mis": parity_mis,
        "ex_mask": ex_mask,
        "rating": y,
    }


with pm.Model() as model:
    ratio_mis = pm.Data("ratio_mis", np.zeros((1, K), dtype="float64"))
    factor_mis = pm.Data("factor_mis", np.zeros((1, K), dtype="float64"))
    digit_mis = pm.Data("digit_mis", np.zeros((1, K), dtype="float64"))
    parity_mis = pm.Data("parity_mis", np.zeros((1, K), dtype="float64"))
    ex_mask = pm.Data("ex_mask", np.ones((1, K), dtype="float64"))

    # Log-similarity cost of each missing relation (attention to that relation).
    w_ratio = pm.HalfNormal("w_ratio", sigma=2.0)
    w_factor = pm.HalfNormal("w_factor", sigma=2.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_parity = pm.HalfNormal("w_parity", sigma=2.0)
    # Criterion (log scale) that average similarity is compared against.
    log_k = pm.Normal("log_k", mu=-1.0, sigma=2.0)
    # Lapse: answer by a coin flip without consulting the examples.
    lapse = pm.Beta("lapse", alpha=1.5, beta=8.0)

    log_sim = -(w_ratio * ratio_mis + w_factor * factor_mis
                + w_digit * digit_mis + w_parity * parity_mis)
    n_ex = pt.sum(ex_mask, axis=1)
    S = pt.sum(pt.exp(log_sim) * ex_mask, axis=1) / n_ex
    p_core = S / (S + pt.exp(log_k))
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(0.5 * lapse + (1.0 - lapse) * p_core, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
