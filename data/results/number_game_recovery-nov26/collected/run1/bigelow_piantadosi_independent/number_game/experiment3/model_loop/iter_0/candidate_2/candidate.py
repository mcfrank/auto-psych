"""Isolation-weighted exemplar memory.

People do not infer a concept: they store each example as a separate exemplar
and say yes in proportion to the target's average similarity to the stored
examples, where similarity falls off with distance on a compressed mental
number line and with mismatches in final digit and in parity. Exemplars are
not stored equally strongly: an example that shares its final digit with none
of the other examples (the odd-one-out) is remembered more vividly (a von
Restorff isolation effect), so it carries more weight in the comparison and
people generalise to its neighbours more than to equally close neighbours of
the conforming examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # maximum number of examples in a set


def prepare_observed(rows):
    n = len(rows)
    dmag = np.zeros((n, K))
    mdig = np.zeros((n, K))
    mpar = np.zeros((n, K))
    isol = np.zeros((n, K))
    mask = np.zeros((n, K))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")]
        t = int(row["target"])
        m = len(ex)
        for k, e in enumerate(ex[:K]):
            dmag[r, k] = abs(np.log(t) - np.log(e))
            mdig[r, k] = float(t % 10 != e % 10)
            mpar[r, k] = float(t % 2 != e % 2)
            if m > 1:
                shared = sum(1 for j, o in enumerate(ex) if j != k and o % 10 == e % 10)
                isol[r, k] = 1.0 - shared / (m - 1)
            mask[r, k] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"ex_dmag": dmag, "ex_mdig": mdig, "ex_mpar": mpar,
            "ex_isol": isol, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    ex_dmag = pm.Data("ex_dmag", np.zeros((1, K)))
    ex_mdig = pm.Data("ex_mdig", np.zeros((1, K)))
    ex_mpar = pm.Data("ex_mpar", np.zeros((1, K)))
    ex_isol = pm.Data("ex_isol", np.zeros((1, K)))
    ex_mask = pm.Data("ex_mask", np.ones((1, K)))

    # Sensitivities of similarity to each dimension of difference.
    c_mag = pm.HalfNormal("c_mag", sigma=5.0)
    c_dig = pm.HalfNormal("c_dig", sigma=3.0)
    c_par = pm.HalfNormal("c_par", sigma=3.0)
    # Isolation effect: extra log memory strength of an isolated exemplar.
    beta_isol = pm.Normal("beta_isol", mu=0.0, sigma=2.0)
    # Response criterion (Luce choice against a fixed criterion, in log units).
    alpha = pm.Normal("alpha", mu=1.0, sigma=2.0)

    sim = pt.exp(-(c_mag * ex_dmag + c_dig * ex_mdig + c_par * ex_mpar))
    w = ex_mask * pt.exp(beta_isol * ex_isol)
    S = pt.sum(w * sim, axis=1) / pt.sum(w, axis=1)
    p = pm.math.sigmoid(alpha + pt.log(S + 1e-12))
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
