"""Diagnostic-attention exemplar similarity.

People do not infer a hidden concept. They keep the examples in memory as
exemplars and say yes to a target the more similar it is, on average, to
them (a GCM-style comparison of mean similarity against a fitted criterion). Similarity is
computed over stored number properties (parity, primality, squares, cubes,
powers of 2, multiples of 3-10, final digit, chart row, digits contained) and
magnitude, with
attention shifted to the properties the examples share: a dimension gets
attention in proportion to how surprising the examples' agreement on it is
(examples sharing a rare property such as "prime" draw much attention; sharing
a common one such as "not a square" draws little), and magnitude gets more
attention the more tightly the examples cluster (log of 100 over their range).
The yes probability is a logistic function of the log mean similarity. So after all-prime sets,
non-prime targets mismatch every example on a heavily attended dimension and
are rejected, while other primes are accepted.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)
_prime = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS])
# Stored property dimensions: a value per number 1-100 on each.
DIMS = [NUMS % 2, _prime.astype(int),
        np.isin(NUMS, [k * k for k in range(1, 11)]).astype(int),
        np.isin(NUMS, [k ** 3 for k in range(1, 5)]).astype(int),
        np.isin(NUMS, [2 ** k for k in range(7)]).astype(int)]
DIMS += [(NUMS % m == 0).astype(int) for m in range(3, 11)]
DIMS += [NUMS % 10, NUMS // 10 - (NUMS % 10 == 0)]  # final digit, chart row
DIMS += [np.array([str(d) in str(n) for n in NUMS]).astype(int) for d in range(10)]
DIMS = np.array(DIMS)  # K x 100
BASE = [{v: np.mean(row == v) for v in np.unique(row)} for row in DIMS]
MAXE = 4


def _features(ex, t):
    """Attention-weighted property mismatch and scaled distance to each example."""
    n = len(ex)
    att = np.zeros(len(DIMS))
    for k, row in enumerate(DIMS):
        vals = [row[x - 1] for x in ex]
        v = max(set(vals), key=vals.count)
        att[k] = (vals.count(v) - 1) * -np.log(BASE[k][v])  # surprise of the agreement
    spread = max(ex) - min(ex) + 10.0
    mis = np.zeros(MAXE)
    dist = np.zeros(MAXE)
    mask = np.zeros(MAXE)
    for i, x in enumerate(ex[:MAXE]):
        mis[i] = np.sum(att * (DIMS[:, t - 1] != DIMS[:, x - 1]))
        dist[i] = abs(t - x) / 10.0 * np.log(100.0 / (spread - 9.0))
        mask[i] = 1.0
    return mis, dist, mask


def prepare_observed(rows):
    cache = {}
    M, Dd, K = [], [], []
    for r in rows:
        key = (str(r["number_set"]).strip(), int(float(r["target"])))
        if key not in cache:
            ex = [int(x) for x in key[0].split(",")]
            cache[key] = _features(ex, key[1])
        m, d, k = cache[key]
        M.append(m); Dd.append(d); K.append(k)
    return {
        "mismatch": np.array(M), "distance": np.array(Dd), "ex_mask": np.array(K),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, MAXE)))
    distance = pm.Data("distance", np.zeros((1, MAXE)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXE)))

    attention_gain = pm.HalfNormal("attention_gain", sigma=2.0)  # weight on property mismatch
    magnitude_weight = pm.HalfNormal("magnitude_weight", sigma=5.0)  # weight on scaled distance
    log_criterion = pm.Normal("log_criterion", -3.0, 3.0)  # similarity criterion for "yes"
    lapse = pm.Beta("lapse", 1.0, 10.0)

    sim = pt.exp(-attention_gain * mismatch - magnitude_weight * distance) * ex_mask
    S = sim.sum(1) / ex_mask.sum(1)  # mean similarity to the stored examples
    g = pm.HalfNormal("response_gain", 3.0)  # steepness of the yes criterion
    p_raw = pt.sigmoid(g * (pt.log(S) - log_criterion))
    p_yes = pm.Deterministic("p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
