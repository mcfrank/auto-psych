"""Normalised-attention-budget similarity.

People judge a target by its similarity to the examples, and they have a fixed
budget of attention that they split across ways the examples could be alike. The
budget goes to each property all the examples share (parity, primality, squares,
cubes, powers of 2, multiples of 3-12, final digit, hundred-chart row, repeated
digits, a digit contained): rarer properties get more, and so do properties
shared by more examples. The remainder goes to position on the number line,
more when the examples cluster tightly. A target's
psychological distance is the attention-weighted share of the budget it
violates (lacking a shared property, or lying outside the examples' span, scaled
by how far outside), and yes-probability falls off with that distance. Because
attention is normalised, a set with a striking shared property makes magnitude
nearly irrelevant, while a set sharing nothing rare is judged by nearness.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)
_prime = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS])
DIMS = [NUMS % 2, _prime.astype(int),
        np.isin(NUMS, [k * k for k in range(1, 11)]).astype(int),
        np.isin(NUMS, [1, 8, 27, 64]).astype(int),
        np.isin(NUMS, [2 ** k for k in range(7)]).astype(int)]
DIMS += [(NUMS % m == 0).astype(int) for m in range(3, 13)]
DIMS += [NUMS % 10, (NUMS - 1) // 10,  # final digit, hundred-chart row
         np.isin(NUMS, [11 * k for k in range(1, 10)]).astype(int)]  # repdigits
DIMS += [np.array([str(d) in str(n) for n in NUMS]).astype(int) for d in range(10)]
DIMS = np.array(DIMS)
BASE = [{v: np.mean(row == v) for v in np.unique(row)} for row in DIMS]


def _features(ex, t):
    """Property salience shared by the examples, the part the target violates,
    and the target's distance outside the examples' span."""
    n = len(ex)
    shared, violated = 0.0, 0.0
    if n >= 2:
        for k, row in enumerate(DIMS):
            vals = {row[x - 1] for x in ex}
            if len(vals) == 1:
                v = vals.pop()
                s = -np.log(BASE[k][v]) * (n - 1)  # surprise of the agreement
                shared += s
                if row[t - 1] != v:
                    violated += s
    lo, hi = min(ex), max(ex)
    return (shared, violated, max(lo - t, t - hi, 0) / 10.0,
            np.log1p((hi - lo) / 10.0), float(n))


def prepare_observed(rows):
    cache = {}
    cols = []
    for r in rows:
        key = (str(r["number_set"]).strip(), int(float(r["target"])))
        if key not in cache:
            cache[key] = _features([int(x) for x in key[0].split(",")], key[1])
        cols.append(cache[key])
    a = np.array(cols, dtype="float64")
    return {
        "shared_salience": a[:, 0], "violated_salience": a[:, 1],
        "outside_span": a[:, 2], "log_spread": a[:, 3], "set_size": a[:, 4],
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared_salience = pm.Data("shared_salience", np.zeros(1))
    violated_salience = pm.Data("violated_salience", np.zeros(1))
    outside_span = pm.Data("outside_span", np.zeros(1))
    log_spread = pm.Data("log_spread", np.zeros(1))
    set_size = pm.Data("set_size", np.ones(1))

    # Salience magnitude claims per example, and how much clustering boosts it.
    log_mag_salience = pm.Normal("log_mag_salience", 1.0, 1.5)
    spread_falloff = pm.HalfNormal("spread_falloff", 2.0)
    log_mag_gradient = pm.Normal("log_mag_gradient", -1.0, 1.5)  # steepness outside span
    intercept = pm.Normal("intercept", 1.0, 2.0)
    gain = pm.HalfNormal("gain", 5.0)  # how fast yes falls with distance

    mag_sal = pt.exp(log_mag_salience - spread_falloff * log_spread) * set_size
    distance = (violated_salience + mag_sal * pt.exp(log_mag_gradient) * outside_span) / (
        shared_salience + mag_sal)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pt.sigmoid(intercept - gain * distance), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
