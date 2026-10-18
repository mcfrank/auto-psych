"""Joint (multiplicative) evidence accumulation over examples.

People treat each example as independent evidence and combine it jointly, as a
likelihood product, rather than averaging resemblance: every example whose
final digit the target does not share costs a fixed amount of log-credibility
(smaller for a same-parity mismatch, larger for an opposite-parity one), and
the costs accumulate with the number of examples (inferred degree between
averaging and full summing), so judgments sharpen with more examples. A
target breaking a final digit shared by three examples is rejected far more
firmly than one breaking a single example's digit. Decade-band familiarity and
common-factor membership still shift the answer as in the incumbent.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n_par = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10)
    n_opp = sum(1 for x in xs if x % 2 != t % 2)
    g = reduce(math.gcd, xs)
    divisible = 1.0 if (g >= 3 and t % g == 0) else 0.0
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    return {
        "n_examples": float(len(xs)),
        "n_parity_mismatch": float(n_par),
        "n_opposite_parity": float(n_opp),
        "cf_multiple": divisible,
        "band_gap_decades": float(max(lo - td, 0, td - hi)),
    }


with pm.Model() as model:
    n_ex = pm.Data("n_examples", np.ones(1, dtype="float64"))
    n_par = pm.Data("n_parity_mismatch", np.zeros(1, dtype="float64"))
    n_opp = pm.Data("n_opposite_parity", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("cf_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("band_gap_decades", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    # Per-example log-credibility costs: same-parity digit mismatch, and the
    # extra cost of an opposite-parity mismatch.
    cost_digit = pm.HalfNormal("cost_digit", sigma=2.0)
    cost_parity_extra = pm.HalfNormal("cost_parity_extra", sigma=2.0)
    # Accumulation: gamma = 1 sums costs across examples (full joint
    # evidence); gamma = 0 would reduce to averaging.
    gamma = pm.Beta("accumulation", alpha=2.0, beta=2.0)
    scale = n_ex ** (gamma - 1.0)
    joint_evidence = scale * (
        -cost_digit * n_par - (cost_digit + cost_parity_extra) * n_opp
    )

    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + joint_evidence + factor_boost * cf_mult + band_boost * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
