"""Edge-salient ideal alternation.

People judge a sequence's randomness by how often it switches between heads and
tails compared with their own personal ideal switching rate, but transitions
near the start and end of the sequence are more salient (serial-position
effect) than those in the middle, so a streak at an edge counts more against a
sequence than the same streak buried in the middle.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_TRANSITIONS = 7  # sequences have at most 8 flips


def compute_features(sequence_a, sequence_b):
    """Per-transition switch indicators, edge distances and validity masks."""

    def per_transition(seq):
        seq = seq.strip().upper()
        n_tr = len(seq) - 1
        if n_tr < 1 or n_tr > MAX_TRANSITIONS:
            raise ValueError(f"unsupported sequence length: {seq!r}")
        sw, dist, mask = [], [], []
        for k in range(MAX_TRANSITIONS):
            if k < n_tr:
                sw.append(1.0 if seq[k] != seq[k + 1] else 0.0)
                dist.append(float(min(k, n_tr - 1 - k)))
                mask.append(1.0)
            else:
                sw.append(0.0)
                dist.append(0.0)
                mask.append(0.0)
        return sw, dist, mask

    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        sw, dist, mask = per_transition(seq)
        for k in range(MAX_TRANSITIONS):
            out[f"sw_{tag}_{k}"] = sw[k]
            out[f"dist_{tag}_{k}"] = dist[k]
            out[f"mask_{tag}_{k}"] = mask[k]
    return out


with pm.Model() as model:
    cols = {}
    for tag in ("a", "b"):
        for kind in ("sw", "dist", "mask"):
            cols[(kind, tag)] = pt.stack(
                [
                    pm.Data(f"{kind}_{tag}_{k}", np.zeros(1, dtype="float64"))
                    for k in range(MAX_TRANSITIONS)
                ],
                axis=1,
            )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Sensitivity to squared distance from the ideal.
    beta = pm.LogNormal("beta", mu=2.5, sigma=0.5)
    # Edge salience: per step inward from the nearest edge, a transition's weight
    # falls by a factor exp(-kappa) (kappa > 0: edges dominate).
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.0)

    def weighted_alt(tag):
        w = pt.exp(-kappa * cols[("dist", tag)]) * cols[("mask", tag)]
        return pt.sum(w * cols[("sw", tag)], axis=1) / pt.sum(w, axis=1)

    alt_a = weighted_alt("a")
    alt_b = weighted_alt("b")
    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(alt_a - theta)
    score_b = -beta * pt.sqr(alt_b - theta)
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(score_a - score_b), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
