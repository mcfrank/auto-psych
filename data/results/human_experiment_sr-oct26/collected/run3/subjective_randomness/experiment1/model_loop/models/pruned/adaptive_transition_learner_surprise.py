"""Adaptive transition-learner surprise.

People read each sequence flip by flip while an online pattern learner predicts
whether the next flip will repeat or switch, starting from a prior expectation
(possibly favouring switches) and updating it from the transitions seen so far.
A sequence looks random to the extent the learner stays surprised (high mean
surprisal per transition); people differ in how decisively this drives choice.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # sequences have at most 8 flips -> 7 transitions


def compute_features(sequence_a, sequence_b):
    def transitions(seq):
        seq = seq.strip().upper()
        return [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]

    ta, tb = transitions(sequence_a), transitions(sequence_b)
    n = len(ta)
    feats = {}
    for t in range(MAX_T):
        feats[f"alt_a_{t}"] = ta[t] if t < n else 0.0
        feats[f"alt_b_{t}"] = tb[t] if t < n else 0.0
        feats[f"valid_{t}"] = 1.0 if t < n else 0.0
    return feats


def _mean_surprisal(alt, valid, c, theta):
    # alt, valid: (trials, MAX_T). Counts of switches/transitions before each step.
    n_alt_before = pt.cumsum(alt, axis=1) - alt
    n_before = pt.cumsum(valid, axis=1) - valid
    p_alt = (c * theta + n_alt_before) / (c + n_before)
    p_obs = pt.clip(alt * p_alt + (1.0 - alt) * (1.0 - p_alt), 1e-6, 1.0)
    surpr = -pt.log(p_obs) * valid
    return pt.sum(surpr, axis=1) / pt.maximum(pt.sum(valid, axis=1), 1.0)


with pm.Model() as model:
    alt_a = pt.stack([pm.Data(f"alt_a_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    alt_b = pt.stack([pm.Data(f"alt_b_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    valid = pt.stack([pm.Data(f"valid_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Prior expectation that the next flip switches, and its strength in pseudo-transitions.
    theta = pm.Beta("theta", alpha=4.0, beta=4.0)
    c = pm.LogNormal("c", mu=np.log(2.0), sigma=0.6)

    # Person-specific decisiveness (non-centred, log-normal population).
    mu_lb = pm.Normal("mu_log_beta", mu=np.log(3.0), sigma=0.7)
    sd_lb = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z_beta", 0.0, 1.0, shape=400)
    beta = pm.Deterministic("beta", pt.exp(mu_lb + sd_lb * z))

    s_a = _mean_surprisal(alt_a, valid, c, theta)
    s_b = _mean_surprisal(alt_b, valid, c, theta)

    # Pairs with identical surprisal profiles differ only by float rounding: treat as exact ties.
    diff = s_a - s_b
    diff = pt.switch(pt.lt(pt.abs(diff), 1e-9), 0.0, diff)

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta[participant_id] * diff), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
