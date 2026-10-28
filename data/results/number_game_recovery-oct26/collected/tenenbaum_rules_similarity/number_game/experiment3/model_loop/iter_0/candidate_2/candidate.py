"""Information-weighted prototype matching.

People do not infer a rule; they abstract a single prototype from the examples
-- the examples' mean magnitude and spread, plus the proportion of examples that
have each familiar number property (even, multiple of 3/4/5/10, prime, square,
cube, power of two, final digit) -- and say "yes" in proportion to how closely
the target matches that prototype. Magnitude distance is judged relative to the
examples' spread (exponential similarity in spread-scaled distance); missing a property the examples share
costs more the rarer that property is among 1-100 (its information content).
Atypical examples (e.g. 100 after 31,19,49) are themselves often rejected.
"""
import math

import numpy as np
import pymc as pm

_N = np.arange(1, 101)
_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}


def _props(n):
    v = [n % 2 == 0, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
         n in _PRIMES, round(n ** 0.5) ** 2 == n, round(n ** (1 / 3)) ** 3 == n,
         (n & (n - 1)) == 0]
    v += [n % 10 == d for d in range(10)]
    return np.array(v, dtype=float)


_TABLE = np.stack([_props(int(n)) for n in _N])           # 100 x F
_BASE = _TABLE.mean(axis=0)
_W_HAS = -np.log(_BASE)              # cost of lacking a property examples have
_W_LACK = -np.log(1.0 - _BASE)       # cost of having a property examples lack


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    proto = np.mean([_TABLE[x - 1] for x in ex], axis=0)
    tf = _TABLE[t - 1]
    feat = float(np.sum(proto * (1 - tf) * _W_HAS + (1 - proto) * tf * _W_LACK))
    m = float(np.mean(ex))
    var = float(np.var(ex))
    return {"proto_sqdev": (t - m) ** 2 / 100.0,
            "proto_var": var / 100.0,
            "proto_feat_dist": feat}


with pm.Model() as model:
    sqdev = pm.Data("proto_sqdev", np.zeros(1, dtype="float64"))
    var = pm.Data("proto_var", np.zeros(1, dtype="float64"))
    feat = pm.Data("proto_feat_dist", np.zeros(1, dtype="float64"))

    # baseline magnitude spread (in units of 10) added to the examples' variance
    tau = pm.LogNormal("tau", mu=0.0, sigma=1.0)
    # weight on property mismatch (nats of distinctiveness)
    c_feat = pm.HalfNormal("c_feat", sigma=1.0)
    hi = pm.Beta("hi", alpha=8.0, beta=2.0)
    lo = pm.Beta("lo", alpha=1.0, beta=9.0)

    c_mag = pm.HalfNormal("c_mag", sigma=2.0)
    log_sim = -c_mag * pm.math.sqrt(sqdev / (var + tau ** 2) + 1e-6) - c_feat * feat
    p = lo + (hi - lo) * pm.math.exp(log_sim)
    p_yes = pm.Deterministic("p_yes", pm.math.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
