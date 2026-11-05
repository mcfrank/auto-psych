"""Rule-uncertainty guessing over exemplar similarity.

People judge a target by summed similarity to the examples (magnitude, final
digit, parity, and the set's common divisor), and turn that similarity into a
yes/no with a fixed criterion. The one decision-level mechanism: how often they
actually consult that judgment depends on whether the examples reveal a clear
rule. When the examples share no suspicious property (common final digit,
multiples of 5, common divisor, shared parity) people are unsure what the
program does and on many trials simply guess, saying yes at their own base
rate; the clearer the shared rule, the less they guess. So rule-less sets are
endorsed at a broad, flat background rate while rule-coherent sets reject
dissimilar targets consistently.
"""
from collections import Counter
from functools import reduce
from math import gcd, log

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _rule_clarity(ex):
    """Suspiciousness of the most specific property all examples share."""
    n = len(ex)
    if n < 2:
        return 0.0
    best = 0.0
    if len({e % 2 for e in ex}) == 1:
        best = max(best, (n - 1) * log(2))
    if all(e % 5 == 0 for e in ex):
        best = max(best, (n - 1) * log(5))
    if len({e % 10 for e in ex}) == 1:
        best = max(best, (n - 1) * log(10))
    g = reduce(gcd, ex)
    if g >= 3:
        best = max(best, (n - 1) * log(min(g, 20)))
    return best / (3 * log(10))


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    n = len(ex)
    g = reduce(gcd, ex)
    out = {}
    for k in range(MAXK):
        if k < n:
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"par_diff_{k}"] = 0.0 if t % 2 == ex[k] % 2 else 1.0
            out[f"off_divisor_{k}"] = 1.0 if (n > 1 and g >= 3 and t % g != 0) else 0.0
            out[f"present_{k}"] = 1.0
        else:
            for nm in ("dist", "unit_diff", "par_diff", "off_divisor", "present"):
                out[f"{nm}_{k}"] = 0.0
    out["rule_clarity"] = _rule_clarity(ex)
    return out


def _stack(name):
    return pt.stack([pm.Data(f"{name}_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)


with pm.Model() as model:
    dist = _stack("dist")
    udiff = _stack("unit_diff")
    pdiff = _stack("par_diff")
    offd = _stack("off_divisor")
    present = _stack("present")
    clarity = pm.Data("rule_clarity", np.zeros(1, dtype="float64"))

    # Simple similarity inference (fixed attention weights).
    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    c_unit = pm.HalfNormal("c_unit", sigma=3.0)
    c_par = pm.HalfNormal("c_par", sigma=2.0)
    c_divisor = pm.HalfNormal("c_divisor", sigma=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    # Decision rule: guessing when no clear rule is visible.
    guess_base = pm.Normal("guess_base", mu=-1.0, sigma=1.5)
    clarity_drop = pm.HalfNormal("clarity_drop", sigma=5.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)

    log_sim = -pt.exp(log_c_mag) * dist - c_unit * udiff - c_par * pdiff - c_divisor * offd
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()
    p_judge = pm.math.sigmoid(alpha + beta * log_summed)

    guess_w = pm.math.sigmoid(guess_base - clarity_drop * clarity)
    p = (1.0 - guess_w) * p_judge + guess_w * guess_yes
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
