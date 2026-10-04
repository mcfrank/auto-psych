"""Balance-restoring tally coin.

People read a sequence flip by flip while keeping a running tally of how far
heads lead tails, and expect a random coin to be self-correcting at the level
of that tally: the further the tally has drifted from even, the more they
expect the next flip to pull it back toward balance. A sequence looks random
to the extent that its path is probable under this balance-restoring
subjective coin (relative to a fair coin); people choose the sequence with
the higher path probability, differing in how strongly this drives their
choice and with a small habitual left/right lean.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_STEPS = 7  # transitions in a length-8 sequence
N_SLOTS = 400


def _lead_step_products(seq):
    """For each flip after the first: (tally lead before the flip) * (flip's direction)."""
    seq = seq.strip().upper()
    lead = 0
    out = []
    for i, c in enumerate(seq):
        step = 1 if c == "H" else -1
        if i > 0:
            out.append(float(lead * step))
        lead += step
    out += [0.0] * (MAX_STEPS - len(out))  # zero lead*step contributes nothing
    return out


def compute_features(sequence_a, sequence_b):
    ma = _lead_step_products(sequence_a)
    mb = _lead_step_products(sequence_b)
    feats = {}
    for t in range(MAX_STEPS):
        feats[f"ls_a_{t}"] = ma[t]
        feats[f"ls_b_{t}"] = mb[t]
    return feats


with pm.Model() as model:
    ls_a = pt.stack(
        [pm.Data(f"ls_a_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_STEPS)],
        axis=1,
    )
    ls_b = pt.stack(
        [pm.Data(f"ls_b_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_STEPS)],
        axis=1,
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Restoring force: how strongly the current lead pulls the expected next flip back.
    kappa = pm.LogNormal("kappa", mu=0.0, sigma=0.75)

    # Person-specific decisiveness (non-centred, positive).
    # Path scores span tens of nats (a streak of 8 scores about -28), so the
    # typical weight per nat is well below one.
    mu_log_beta = pm.Normal("mu_log_beta", mu=-2.0, sigma=1.0)
    sd_log_beta = pm.HalfNormal("sd_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=N_SLOTS)
    beta = pt.exp(mu_log_beta + sd_log_beta * z_beta)

    # Person-specific left/right lean.
    sd_bias = pm.HalfNormal("sd_bias", sigma=0.5)
    z_bias = pm.Normal("z_bias", 0.0, 1.0, shape=N_SLOTS)
    bias = sd_bias * z_bias

    # Log-likelihood ratio of each path: subjective coin vs fair coin.
    # P(flip) = sigmoid(-kappa * lead * step); at an even tally this is 1/2.
    # Divided by kappa so that beta carries the overall strength and kappa only
    # how much the path matters beyond the final lead (as kappa -> 0 the score
    # tends to minus half the sum of lead*step, a function of the final count).
    score_a = pt.sum(pt.log(2.0) - pt.softplus(kappa * ls_a), axis=1) / kappa
    score_b = pt.sum(pt.log(2.0) - pt.softplus(kappa * ls_b), axis=1) / kappa
    # Padding (lead*step = 0) contributes log(2) + log(1/2) = 0.

    eta = beta[participant_id] * (score_a - score_b) + bias[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
