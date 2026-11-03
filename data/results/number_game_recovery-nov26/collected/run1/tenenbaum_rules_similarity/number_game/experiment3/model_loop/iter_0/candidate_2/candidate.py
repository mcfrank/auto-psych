"""Attention-weighted set prototype (prototype account, no concept inference).

People abstract a single prototype of the example set: its typical magnitude
(mean) and spread (SD) on the number line, plus the profile of familiar number
properties (parity, multiples, final and contained digits, primes, squares,
powers, ...) giving the proportion of examples that have each property. A target
is judged by its similarity to that prototype: attention to a property grows
with how consistently the examples share it (proportion ** gamma) and with its
distinctiveness (log of 100 / property size), and similarity falls with the
attention-weighted share of prototype properties the target lacks and with its
distance from the prototype's magnitude relative to the examples' spread.
"yes" is a logistic function of this similarity.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _properties():
    props = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    props.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    props.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    props.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    props.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return list(props.values())


PROPS = _properties()
HAS = np.array([np.isin(NUMBERS, c) for c in PROPS], dtype="float64")  # (F, 100)
LACKS = 1.0 - HAS
DISTINCT = np.log(100.0 / HAS.sum(axis=1))  # rarer properties are more distinctive

with pm.Model() as model:
    proto_logprop = pm.Data("proto_logprop", np.zeros((1, len(PROPS))))  # log share of examples with each property
    proto_present = pm.Data("proto_present", np.zeros((1, len(PROPS))))  # 1 if any example has it
    proto_offset = pm.Data("proto_offset", np.zeros((1, 100)))  # |target - mean of examples| / 100
    proto_spread = pm.Data("proto_spread", np.zeros((1, 1)), shape=(None, 1))  # SD of examples / 100
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    feature_weight = pm.HalfNormal("feature_weight", sigma=5.0)
    magnitude_weight = pm.HalfNormal("magnitude_weight", sigma=5.0)
    consistency_gamma = pm.LogNormal("consistency_gamma", mu=np.log(2.0), sigma=1.0)
    # How much wider the magnitude tolerance becomes per unit of example spread.
    spread_sensitivity = pm.LogNormal("spread_sensitivity", mu=0.0, sigma=1.0)

    attention = proto_present * pt.exp(consistency_gamma * proto_logprop) * DISTINCT[None, :]
    feature_mismatch = (attention @ pt.as_tensor(LACKS)) / (attention.sum(axis=1, keepdims=True) + 1e-9)
    magnitude_mismatch = proto_offset / (1.0 + spread_sensitivity * proto_spread)
    logit = bias - feature_weight * feature_mismatch - magnitude_weight * magnitude_mismatch
    p_cell = pm.math.sigmoid(logit).reshape((-1,))
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    share = np.stack([HAS[:, np.array(s) - 1].mean(axis=1) for s in unique])
    present = (share > 0).astype("float64")
    logprop = np.where(share > 0, np.log(np.where(share > 0, share, 1.0)), 0.0)
    means = np.array([np.mean(s) for s in unique])
    spread = np.array([np.std(s) for s in unique]) / 100.0
    return {
        "proto_logprop": logprop,
        "proto_present": present,
        "proto_offset": np.abs(NUMBERS[None, :] - means[:, None]) / 100.0,
        "proto_spread": spread[:, None],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
