"""Serial-position (primacy/recency) weighted encoding of alternation.

People encode a coin-flip sequence with serial-position effects: transitions
near the start and end are encoded more strongly than those in the middle, so
the perceived switching rate is dominated by the sequence's edges. Each person
compares this edge-weighted switching rate with their own ideal switching rate,
and guesses on some trials at a personal lapse rate. A streak or rigid
alternation at an edge thus weighs more than the same pattern in the middle.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_TRANS = 7


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    if len(a) < 2 or len(a) != len(b):
        raise ValueError(f"bad pair: {a!r} {b!r}")
    n = len(a) - 1
    out = {}
    for k in range(MAX_TRANS):
        valid = k < n
        out[f"sw_a_{k}"] = float(a[k] != a[k + 1]) if valid else 0.0
        out[f"sw_b_{k}"] = float(b[k] != b[k + 1]) if valid else 0.0
        out[f"mask_{k}"] = 1.0 if valid else 0.0
        # distance of transition k from the nearest edge, relative to length
        out[f"edge_dist_{k}"] = (min(k, n - 1 - k) / max(n - 1, 1)) if valid else 0.0
    return out


with pm.Model() as model:
    sw_a = pt.stack([pm.Data(f"sw_a_{k}", np.zeros(1)) for k in range(MAX_TRANS)], axis=1)
    sw_b = pt.stack([pm.Data(f"sw_b_{k}", np.zeros(1)) for k in range(MAX_TRANS)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{k}", np.zeros(1)) for k in range(MAX_TRANS)], axis=1)
    dist = pt.stack([pm.Data(f"edge_dist_{k}", np.zeros(1)) for k in range(MAX_TRANS)], axis=1)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Serial-position salience: weight decays with distance from the nearest edge.
    edge_decay = pm.HalfNormal("edge_decay", sigma=2.0)
    w = mask * pt.exp(-edge_decay * dist)
    wsum = pt.sum(w, axis=1)
    enc_a = pt.sum(w * sw_a, axis=1) / wsum
    enc_b = pt.sum(w * sw_b, axis=1) / wsum

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.5, sigma=0.5)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(enc_a - theta)
    score_b = -beta * pt.sqr(enc_b - theta)
    lam = lapse[participant_id]
    p_engaged = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
