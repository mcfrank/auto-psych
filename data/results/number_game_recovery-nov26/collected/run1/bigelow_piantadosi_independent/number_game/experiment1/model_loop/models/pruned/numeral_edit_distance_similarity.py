"""Numeral edit-distance similarity.

People judge membership by how similar the target looks, as a written numeral,
to each example: similarity falls off exponentially with the digit edit
(Levenshtein) distance between the target's numeral and an example's, and the
yes probability grows with the target's average similarity across examples.
Arithmetic rules and number-line magnitude play no separate role.
"""
import numpy as np
import pymc as pm


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compute_features(number_set, target):
    """Fraction of examples at numeral edit distance 0, 1, 2 and >=3 from the target."""
    examples = [str(int(x)) for x in number_set.split(",")]
    t = str(int(target))
    dists = [min(_lev(t, e), 3) for e in examples]
    n = float(len(dists))
    return {f"frac_edit{k}": sum(d == k for d in dists) / n for k in range(4)}


with pm.Model() as model:
    f0 = pm.Data("frac_edit0", np.zeros(1, dtype="float64"))
    f1 = pm.Data("frac_edit1", np.zeros(1, dtype="float64"))
    f2 = pm.Data("frac_edit2", np.zeros(1, dtype="float64"))
    f3 = pm.Data("frac_edit3", np.zeros(1, dtype="float64"))

    # Steepness of the similarity gradient per digit edit.
    decay = pm.HalfNormal("decay", sigma=2.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    g = pm.math.exp(-decay)
    sim = f0 + f1 * g + f2 * g ** 2 + f3 * g ** 3
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
