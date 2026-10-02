"""Serial-position-weighted alternation.

People attend unevenly to a coin-flip sequence: its beginning and end are more
salient than its middle (a serial-position effect), so the switches and repeats
between adjacent flips at the edges weigh more in their impression of how often
the sequence switches. Each person compares this salience-weighted switching
rate with their own ideal switching rate and picks the sequence that comes
closer, so a streak at the start or end of a sequence looks less random than
the same streak in the middle.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_TRANSITIONS = 7  # sequences of up to 8 flips


def compute_features(sequence_a, sequence_b):
    def transitions(seq):
        seq = seq.strip().upper()
        n = len(seq) - 1
        if n < 1 or n > MAX_TRANSITIONS:
            raise ValueError(f"unsupported sequence length: {seq!r}")
        switch = [0.0] * MAX_TRANSITIONS
        mask = [0.0] * MAX_TRANSITIONS
        edge = [0.0] * MAX_TRANSITIONS
        centre = (n - 1) / 2.0
        for i in range(n):
            switch[i] = 1.0 if seq[i] != seq[i + 1] else 0.0
            mask[i] = 1.0
            # 1 at the first/last transition, 0 at the middle.
            edge[i] = abs(i - centre) / centre if centre > 0 else 1.0
        return switch, mask, edge

    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        s, m, e = transitions(seq)
        for k in range(MAX_TRANSITIONS):
            out[f"sw_{tag}_{k}"] = s[k]
            out[f"mask_{tag}_{k}"] = m[k]
            out[f"edge_{tag}_{k}"] = e[k]
    return out


def _stack(prefix, tag):
    cols = [
        pm.Data(f"{prefix}_{tag}_{k}", np.zeros(1, dtype="float64"))
        for k in range(MAX_TRANSITIONS)
    ]
    return pt.stack(cols, axis=1)


with pm.Model() as model:
    sw_a, mask_a, edge_a = _stack("sw", "a"), _stack("mask", "a"), _stack("edge", "a")
    sw_b, mask_b, edge_b = _stack("sw", "b"), _stack("mask", "b"), _stack("edge", "b")
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Edge salience: log weight of an edge transition relative to the middle.
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.0)

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.0, sigma=1.0)

    def weighted_alt(sw, mask, edge):
        w = mask * pt.exp(kappa * edge)
        return pt.sum(w * sw, axis=1) / pt.sum(w, axis=1)

    alt_a = weighted_alt(sw_a, mask_a, edge_a)
    alt_b = weighted_alt(sw_b, mask_b, edge_b)

    theta = ideal[participant_id]
    score_a = -pt.sqr(alt_a - theta)
    score_b = -pt.sqr(alt_b - theta)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
