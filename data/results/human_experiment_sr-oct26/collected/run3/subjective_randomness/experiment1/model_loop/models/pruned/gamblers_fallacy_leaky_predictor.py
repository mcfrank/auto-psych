"""Gambler's-fallacy leaky predictor.

People judge randomness by running a gambler's-fallacy predictor through the
sequence: before each flip they expect the coin to correct the heads/tails
imbalance seen so far, with recent flips weighing more than older ones in a
leaky running tally. A sequence looks random to the extent its flips match
those corrective expectations (high average log predictive probability), so
the order of flips matters: a late repeat or streak, violating a just-built
expectation of reversal, is penalised more than the same pattern early on.
People differ in how sharply they apply the judgment (person-level
sensitivity).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_LEN = 8
N_SLOTS = 400


def _encode(seq):
    s = seq.strip().upper()
    x = np.zeros(MAX_LEN)
    for i, c in enumerate(s):
        x[i] = 1.0 if c == "H" else -1.0
    return x, len(s)


def compute_features(sequence_a, sequence_b):
    xa, na = _encode(sequence_a)
    xb, nb = _encode(sequence_b)
    feats = {}
    for i in range(MAX_LEN):
        feats[f"xa_{i}"] = float(xa[i])
        feats[f"xb_{i}"] = float(xb[i])
    feats["len_a"] = float(na)
    feats["len_b"] = float(nb)
    return feats


def _score(xs, length, lam, g):
    # xs: list of MAX_LEN per-trial +-1 vectors (0 padding). Flip t (1 <= t < length)
    # is predicted from the leaky H-minus-T tally of flips before it:
    # tally_t = lam * tally_{t-1} + x_{t-1}; P(actual flip) = sigmoid(-g * tally_t * x_t).
    total = 0.0
    tally = 0.0
    for t in range(1, MAX_LEN):
        tally = lam * tally + xs[t - 1]
        log_pred = -pt.softplus(g * tally * xs[t])
        total = total + log_pred * pt.cast(pt.lt(t, length), "float64")
    return total / pt.maximum(length - 1.0, 1.0)


with pm.Model() as model:
    xa = [pm.Data(f"xa_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_LEN)]
    xb = [pm.Data(f"xb_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_LEN)]
    len_a = pm.Data("len_a", np.zeros(1, dtype="float64"))
    len_b = pm.Data("len_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Strength of the expected correction per unit of tally.
    g = pm.LogNormal("g", mu=0.0, sigma=0.75)
    # Memory leak: 0 = only the last flip matters, 1 = full running balance.
    lam = pm.Beta("lam", alpha=2.0, beta=2.0)

    # Person-level sensitivity (non-centred, log scale).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.75)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta_p = pt.exp(mu_log_beta + sigma_log_beta * z)

    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    score_a = _score(xa, len_a, lam, g)
    score_b = _score(xb, len_b, lam, g)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta_p[participant_id] * (score_a - score_b) + side_bias),
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
