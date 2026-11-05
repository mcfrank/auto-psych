"""Upward-directed similarity gradient.

People judge a target by its similarity to the examples on a single explicit
metric: the target loses similarity for each number property the examples
consistently share (parity, final digit, digits contained, multiples, special
kinds such as squares or primes) that it lacks, with a property's attention
rising steeply with how consistently the examples share it, and for how far it
lies below the smallest example on the number line. The magnitude gradient is
one-directional: numbers above the largest example lose no similarity for
their magnitude, numbers below the smallest lose more the farther down they
lie, so people generalize upward more than downward.
"""
import numpy as np
import pymc as pm

_N = np.arange(1, 101)


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _property_table():
    props = [_N % 2 == 1, _N % 2 == 0]
    props.append(np.array([_isprime(n) for n in _N]))
    props.append(np.isin(_N, [k * k for k in range(1, 11)]))
    props.append(np.isin(_N, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        props.append(np.isin(_N, [b ** k for k in range(1, 8)]))
    for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        props.append(_N % m == 0)
    for d in range(10):
        props.append(_N % 10 == d)
    for d in range(10):
        props.append(np.array([str(d) in str(n) for n in _N]))
    return np.array(props, dtype="float64").T  # (100, K)


_PROPS = _property_table()
_K = _PROPS.shape[1]


def prepare_observed(rows):
    sets, set_idx, tprops, below, ratings = {}, [], [], [], []
    for r in rows:
        ns = str(r["number_set"]).strip()
        if ns not in sets:
            sets[ns] = len(sets)
        ex = [int(x) for x in ns.split(",")]
        t = int(float(r["target"]))
        set_idx.append(sets[ns])
        tprops.append(_PROPS[t - 1])
        below.append(max(min(ex) - t, 0) / 100.0)
        ratings.append(int(float(r["rating"])))
    share = np.zeros((len(sets), _K))
    for ns, i in sets.items():
        ex = np.array([int(x) for x in ns.split(",")])
        share[i] = _PROPS[ex - 1].mean(axis=0)
    return {
        "example_share": share,
        "set_index": np.array(set_idx, dtype="int64"),
        "target_props": np.array(tprops, dtype="float64"),
        "distance_below": np.array(below, dtype="float64"),
        "rating": np.array(ratings, dtype="int64"),
    }


with pm.Model() as model:
    example_share = pm.Data("example_share", np.full((1, _K), 0.5))
    set_index = pm.Data("set_index", np.zeros(1, dtype="int64"))
    target_props = pm.Data("target_props", np.zeros((1, _K)))
    distance_below = pm.Data("distance_below", np.zeros(1))

    # Endorsement of a target maximally similar to the examples.
    intercept = pm.Normal("intercept", mu=1.0, sigma=1.5)
    # Similarity lost per unit of attended property mismatch.
    feature_sens = pm.LogNormal("feature_sens", mu=1.0, sigma=1.0)
    # How steeply attention to a property rises with how consistently it is shared.
    consistency_power = pm.LogNormal("consistency_power", mu=1.0, sigma=0.7)
    # Similarity lost per 100 units below the smallest example (none above).
    downward_sens = pm.HalfNormal("downward_sens", sigma=3.0)

    attention = (example_share + 1e-9) ** consistency_power  # (sets, K)
    att = attention[set_index]
    mismatch = (att * (1.0 - target_props)).sum(axis=1) / att.sum(axis=1)

    logit_p = intercept - feature_sens * mismatch - downward_sens * distance_below
    p_yes = pm.Deterministic(
        "p_yes", pm.math.clip(pm.math.sigmoid(logit_p), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
