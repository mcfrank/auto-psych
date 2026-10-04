"""Position-weighted switch impression.

People form their impression of how often the coin switches mainly from one end
of the sequence (recency or primacy, a fitted direction and strength), and judge
a sequence as more random the closer that position-weighted switch rate is to a
shared ideal switch rate for a random coin.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # transitions in a length-8 sequence


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        seq = seq.strip().upper()
        n = len(seq) - 1
        for t in range(MAX_T):
            if t < n:
                sw = 1.0 if seq[t] != seq[t + 1] else 0.0
                pos = 0.0 if n == 1 else -1.0 + 2.0 * t / (n - 1)
                m = 1.0
            else:
                sw, pos, m = 0.0, 0.0, 0.0
            out[f"sw_{tag}_{t}"] = sw
            out[f"pos_{tag}_{t}"] = pos
            out[f"m_{tag}_{t}"] = m
    return out


with pm.Model() as model:
    data = {}
    for tag in ("a", "b"):
        for kind in ("sw", "pos", "m"):
            data[(kind, tag)] = pt.stack(
                [pm.Data(f"{kind}_{tag}_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)],
                axis=1,
            )

    # Direction/strength of positional weighting: >0 recency, <0 primacy.
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)
    # Shared ideal switch rate.
    rho = pm.Beta("rho", alpha=2.0, beta=2.0)
    # Sensitivity to squared distance from the ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)

    def weighted_rate(tag):
        w = pt.exp(gamma * data[("pos", tag)]) * data[("m", tag)]
        return (w * data[("sw", tag)]).sum(axis=1) / w.sum(axis=1)

    r_a = weighted_rate("a")
    r_b = weighted_rate("b")
    score_diff = (r_b - rho) ** 2 - (r_a - rho) ** 2
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
