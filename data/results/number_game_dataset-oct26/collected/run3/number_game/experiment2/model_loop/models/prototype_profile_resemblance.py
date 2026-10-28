"""Prototype resemblance: people abstract one prototype from the examples.

People do not infer a concept; they abstract a single prototype from the
examples — a central magnitude (the examples' mean, with a tolerance set by
their spread) and a property profile (the fraction of examples having each
familiar number property: parity, multiple-of-k, final digit, contained digit,
decade, prime, square, cube, power-of-k) — and say "yes" to the degree the
target resembles it. Resemblance falls with the target's distance from the
prototype's centre and with each prototype property the target lacks (weighted
by the property's fraction in the prototype and by its rarity in 1-100, with an
inferred rarity exponent); the property profile is trusted more the more
examples formed it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
_PRIMES = [n for n in NUMBERS if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def _properties():
    props = [NUMBERS % 2 == 0, NUMBERS % 2 == 1]
    props += [NUMBERS % k == 0 for k in range(3, 13)]
    props += [NUMBERS % 10 == d for d in range(10)]
    props += [np.array([str(d) in str(n) for n in NUMBERS]) for d in range(10)]
    props += [np.minimum(NUMBERS // 10, 9) == t for t in range(10)]
    props.append(np.isin(NUMBERS, _PRIMES))
    props.append(np.isin(NUMBERS, [k * k for k in range(1, 11)]))
    props.append(np.isin(NUMBERS, [1, 8, 27, 64]))
    for b in range(2, 11):
        props.append(np.isin(NUMBERS, [b ** e for e in range(1, 8)]))
    return np.array(props, dtype="float64")  # (F, 100)


HAS = _properties()
RARITY = np.log(100.0 / HAS.sum(axis=1))  # information carried by each property
N_PROP = HAS.shape[0]


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    missing, mag_dev, spread, size = [], [], [], []
    for s, t in cells:
        ex = np.array(s) - 1
        profile = HAS[:, ex].mean(axis=1)  # fraction of examples with each property
        missing.append(profile * (1.0 - HAS[:, t - 1]))
        mag_dev.append(abs(t - np.mean(s)) / 100.0)
        spread.append(np.std(s) / 100.0)
        size.append(float(len(s)))
    return {
        "missing": np.array(missing, dtype="float64"),
        "mag_dev": np.array(mag_dev, dtype="float64"),
        "spread": np.array(spread, dtype="float64"),
        "set_size": np.array(size, dtype="float64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    missing = pm.Data("missing", np.zeros((1, N_PROP)))  # (cells, F)
    mag_dev = pm.Data("mag_dev", np.zeros(1))
    spread = pm.Data("spread", np.zeros(1))
    set_size = pm.Data("set_size", np.ones(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    baseline = pm.Normal("baseline", mu=2.0, sigma=2.0)  # log-odds of yes at the prototype
    rarity_exp = pm.LogNormal("rarity_exp", mu=0.0, sigma=0.5)
    feature_weight = pm.LogNormal("feature_weight", mu=0.0, sigma=1.0)
    trust_growth = pm.Normal("trust_growth", mu=0.0, sigma=1.0)
    magnitude_weight = pm.LogNormal("magnitude_weight", mu=1.0, sigma=1.0)
    tolerance = pm.LogNormal("tolerance", mu=np.log(0.1), sigma=0.75)

    feat_dist = pt.dot(missing, pt.as_tensor_variable(RARITY) ** rarity_exp)
    mag_dist = mag_dev / (spread + tolerance)
    logit = (baseline
             - feature_weight * pt.exp(trust_growth * pt.log(set_size)) * feat_dist
             - magnitude_weight * mag_dist)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit)[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
