"""Glimpse evidence accumulation (refines windowed_glimpse_expectation).

People read a coin-flip sequence through a limited working-memory window and
judge it random by how well the H/T mix of its glimpses matches what they expect
from a random coin (given their own belief about how often a coin switches). The
one change: evidence of non-randomness accumulates over glimpses (summed, not
averaged), so a longer sequence -- more glimpses -- is judged more decisively
than a short one with the same average mismatch.

Implementation: for window widths w = 2..6, D_w is the mean squared local
imbalance ((#H - #T) / w)^2 over the n_w = L - w + 1 width-w windows (a sequence
shorter than w is seen whole, n_w = 1). A person believing a coin switches with
probability q expects E_w(q) = (w + 2 * sum_{k<w} (w - k) rho^k) / w^2,
rho = 1 - 2q. The window kernel weight_w is proportional to r^(w - 2) with a
shared retention r. Perceived non-randomness is the kernel-weighted squared
mismatch (D_w - E_w(q))^2 multiplied by the number of glimpses n_w taken at that
width (the accumulation); q is personal (non-centred logit-normal population).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (unique across a run's experiments); an id at
# or beyond it fails loudly at indexing.
MAX_PARTICIPANTS = 400
WIDTHS = (2, 3, 4, 5, 6)


def compute_features(sequence_a, sequence_b):
    def local_imbalance(seq, w):
        seq = seq.strip().upper()
        if len(seq) < 2 or set(seq) - {"H", "T"}:
            raise ValueError(f"bad sequence: {seq!r}")
        x = [1 if c == "H" else -1 for c in seq]
        w = min(w, len(x))
        vals = [(sum(x[i:i + w]) / w) ** 2 for i in range(len(x) - w + 1)]
        return sum(vals) / len(vals), float(w), float(len(vals))

    out = {}
    for w in WIDTHS:
        for side, seq in (("a", sequence_a), ("b", sequence_b)):
            d, eff_w, n_win = local_imbalance(seq, w)
            out[f"imb{w}_{side}"] = d
            out[f"effw{w}_{side}"] = eff_w
            out[f"nwin{w}_{side}"] = n_win
    return out


def expected_imbalance(eff_w, rho):
    """E[(sum of w +/-1 flips / w)^2] under a Markov coin with lag-1 corr rho."""
    total = eff_w
    for k in range(1, max(WIDTHS)):
        total = total + 2.0 * pt.maximum(eff_w - k, 0.0) * rho ** k
    return total / pt.sqr(eff_w)


with pm.Model() as model:
    imb = {}
    effw = {}
    nwin = {}
    for side in ("a", "b"):
        imb[side] = pt.stack(
            [pm.Data(f"imb{w}_{side}", np.zeros(1, dtype="float64")) for w in WIDTHS], axis=1
        )
        effw[side] = pt.stack(
            [pm.Data(f"effw{w}_{side}", np.full(1, float(w))) for w in WIDTHS], axis=1
        )
        nwin[side] = pt.stack(
            [pm.Data(f"nwin{w}_{side}", np.ones(1, dtype="float64")) for w in WIDTHS], axis=1
        )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal believed switch probability of a random coin (logit scale).
    mu_q = pm.Normal("mu_q", mu=0.4, sigma=1.0)
    sigma_q = pm.HalfNormal("sigma_q", sigma=1.0)
    z_q = pm.Normal("z_q", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    q = pm.Deterministic("q", pm.math.sigmoid(mu_q + sigma_q * z_q))

    # Shared memory retention: how much wider glimpses count.
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Sensitivity to accumulated mismatch (per glimpse, so smaller than the
    # averaged model's).
    beta = pm.LogNormal("beta", mu=1.5, sigma=1.0)

    rho = (1.0 - 2.0 * q[participant_id])[:, None]
    expo = pt.as_tensor_variable(np.array(WIDTHS, dtype="float64") - 2.0)[None, :]
    weights = retention ** expo
    weights = weights / pt.sum(weights, axis=1, keepdims=True)

    def score(side):
        mismatch = pt.sqr(imb[side] - expected_imbalance(effw[side], rho))
        return -pt.sum(weights * nwin[side] * mismatch, axis=1)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score("a") - score("b"))))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
