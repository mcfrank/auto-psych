"""Prototype account of number-concept generalization.

People do not infer a rule. They boil the examples down to one prototype that
records how often each familiar number property occurs among the examples and
the examples' average magnitude. A target is endorsed according to its
similarity to that prototype. Each property's weight in the comparison rises
steeply with the share of examples that have it. Similarity falls with the
fraction of that weight on properties the target lacks, and with the target's
distance from the examples' central magnitude.
"""
import numpy as np
import pymc as pm

_N = np.arange(1, 101)


def _property_table():
    props = [_N % 2 == 1, _N % 2 == 0]
    props.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in _N]))
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
    return np.array(props, dtype="float64").T  # (100, n_props)


_PROPS = _property_table()
_K = _PROPS.shape[1]


def prepare_observed(rows):
    sets, set_idx, tprops, mdist, ratings = {}, [], [], [], []
    for r in rows:
        ns = str(r["number_set"]).strip()
        if ns not in sets:
            sets[ns] = len(sets)
        ex = [int(x) for x in ns.split(",")]
        t = int(float(r["target"]))
        set_idx.append(sets[ns])
        tprops.append(_PROPS[t - 1])
        mdist.append(abs(t - np.mean(ex)) / 100.0)
        ratings.append(int(float(r["rating"])))
    share = np.zeros((len(sets), _K))
    for ns, i in sets.items():
        ex = np.array([int(x) for x in ns.split(",")])
        share[i] = _PROPS[ex - 1].mean(axis=0)
    return {
        "proto_share": share,
        "set_index": np.array(set_idx, dtype="int64"),
        "target_props": np.array(tprops, dtype="float64"),
        "centre_distance": np.array(mdist, dtype="float64"),
        "rating": np.array(ratings, dtype="int64"),
    }


with pm.Model() as model:
    proto_share = pm.Data("proto_share", np.full((1, _K), 0.5))
    set_index = pm.Data("set_index", np.zeros(1, dtype="int64"))
    target_props = pm.Data("target_props", np.zeros((1, _K)))
    centre_distance = pm.Data("centre_distance", np.zeros(1))

    # Response bias at perfect similarity to the prototype.
    intercept = pm.Normal("intercept", mu=1.0, sigma=1.5)
    # Sensitivity to missing prototype properties and to magnitude distance.
    feature_sens = pm.HalfNormal("feature_sens", sigma=5.0)
    magnitude_sens = pm.HalfNormal("magnitude_sens", sigma=3.0)
    # How steeply a property's weight grows with the share of examples having it.
    log_consistency_power = pm.Normal("log_consistency_power", mu=0.5, sigma=1.0)
    power = pm.math.exp(log_consistency_power)

    weights = (proto_share + 1e-9) ** power  # (sets, K)
    w_trial = weights[set_index]  # (trials, K)
    missing = (w_trial * (1.0 - target_props)).sum(axis=1) / w_trial.sum(axis=1)

    logit_p = intercept - feature_sens * missing - magnitude_sens * centre_distance
    p_yes = pm.Deterministic("p_yes", pm.math.clip(pm.math.sigmoid(logit_p), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
