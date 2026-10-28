"""Coherence-driven selective-attention exemplar model.

People do not infer a concept; they judge the target by its similarity to the
stored examples (exemplar generalization), but attention is drawn to the
dimensions on which the stored examples cohere: every property all examples
share (parity, final digit, decade, contained digit, prime, square, cube,
power-of-k, multiple-of-k) receives attention in proportion to its
distinctiveness in 1-100 (-log base rate), so a target lacking a rare shared
property is dissimilar to every example. Closeness on the number line counts on
a scale that stretches with the examples' spread. "Yes" is a Luce choice
between the target's mean similarity to the examples and a fixed criterion.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(math.isqrt(n)) + 1))}
POWERS = {k: {k ** e for e in range(1, 8) if k ** e <= 100} for k in range(2, 11)}


def properties(n):
    props = {"even" if n % 2 == 0 else "odd", f"last{n % 10}", f"decade{min(n // 10, 9)}"}
    props |= {f"has{d}" for d in set(str(n))}
    if n in PRIMES:
        props.add("prime")
    if round(n ** 0.5) ** 2 == n:
        props.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        props.add("cube")
    props |= {f"pow{k}" for k, s in POWERS.items() if n in s}
    props |= {f"mult{k}" for k in range(3, 13) if n % k == 0}
    return props


_PROPS = {n: properties(n) for n in range(1, 101)}
_COUNT = {}
for _n in range(1, 101):
    for _p in _PROPS[_n]:
        _COUNT[_p] = _COUNT.get(_p, 0) + 1
RARITY = {p: -math.log(c / 100.0) for p, c in _COUNT.items()}


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")][:MAX_EX]
    t = int(target)
    shared = set.intersection(*(_PROPS[x] for x in ex))
    total = sum(RARITY[p] for p in shared)
    # Attention-weighted mismatch on the shared dimensions; every example has
    # all shared properties, so this mismatch is the same for each exemplar.
    miss = sum(RARITY[p] for p in shared if p not in _PROPS[t])
    out = {"prop_mismatch": miss / total if total > 0 else 0.0,
           "ex_range": (max(ex) - min(ex)) / 100.0}
    for i in range(MAX_EX):
        if i < len(ex):
            out[f"mag_d{i}"] = abs(t - ex[i]) / 100.0
            out[f"mask{i}"] = 1.0
        else:
            out[f"mag_d{i}"] = 0.0
            out[f"mask{i}"] = 0.0
    return out


with pm.Model() as model:
    prop_mismatch = pm.Data("prop_mismatch", np.zeros(1))
    ex_range = pm.Data("ex_range", np.zeros(1))
    mag = pt.stack([pm.Data(f"mag_d{i}", np.zeros(1)) for i in range(MAX_EX)], axis=0)
    mask = pt.stack([pm.Data(f"mask{i}", np.ones(1)) for i in range(MAX_EX)], axis=0)

    # Sensitivity to mismatch on the attended shared properties, baseline
    # width of the number-line gradient (grows with the examples' range), and
    # the Luce response criterion.
    c_prop = pm.HalfNormal("c_prop", sigma=5.0)
    tau = pm.HalfNormal("tau", sigma=0.3)
    log_k = pm.Normal("log_k", mu=-1.0, sigma=1.5)

    width = tau + ex_range + 1e-3
    sim = pt.exp(-c_prop * prop_mismatch[None, :] - mag / width[None, :]) * mask
    S = sim.sum(axis=0) / mask.sum(axis=0)
    k = pt.exp(log_k)
    p_yes = pm.Deterministic("p_yes", pt.clip(S / (S + k), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
