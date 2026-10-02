"""Windowed glimpse expectation.

People read a coin-flip sequence through a limited working-memory window,
holding only the last few flips at a time, and compare how mixed each glimpse is
(its H/T balance) with how mixed they expect a glimpse of a random coin to be; a
sequence looks random to the extent its glimpses match that expectation. The
expectation comes from each person's own belief about how often a random coin
switches sides, so some expect near-perfect mixing and others streakier
glimpses; because the window spans more than two flips, lopsided H/T counts and
long runs are judged beyond what the alternation rate alone shows.

Implementation: for window widths w = 2..6, D_w is the mean squared local
imbalance ((#H - #T) / w)^2 over all width-w windows (a sequence shorter than w
is seen whole). A person believing a coin switches with probability q expects a
width-w glimpse to have squared imbalance
E_w(q) = (w + 2 * sum_{k<w} (w - k) rho^k) / w^2, rho = 1 - 2q
(at w = 2 the comparison reduces to alternation rate vs q). The memory window is
a soft kernel over widths, weight_w proportional to r^(w - 2), with a shared
retention r. Perceived randomness is minus the kernel-weighted squared mismatch
between D_w and E_w(q); q is personal (non-centred logit-normal population).
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
        return sum(vals) / len(vals), float(w)

    out = {}
    for w in WIDTHS:
        for side, seq in (("a", sequence_a), ("b", sequence_b)):
            d, eff_w = local_imbalance(seq, w)
            out[f"imb{w}_{side}"] = d
            out[f"effw{w}_{side}"] = eff_w
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
    for side in ("a", "b"):
        imb[side] = pt.stack(
            [pm.Data(f"imb{w}_{side}", np.zeros(1, dtype="float64")) for w in WIDTHS], axis=1
        )
        effw[side] = pt.stack(
            [pm.Data(f"effw{w}_{side}", np.full(1, float(w))) for w in WIDTHS], axis=1
        )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal believed switch probability of a random coin (logit scale).
    mu_q = pm.Normal("mu_q", mu=0.4, sigma=1.0)
    sigma_q = pm.HalfNormal("sigma_q", sigma=1.0)
    z_q = pm.Normal("z_q", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    q = pm.Deterministic("q", pm.math.sigmoid(mu_q + sigma_q * z_q))

    # Shared memory retention: how much wider glimpses count.
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Sensitivity to mismatch.
    beta = pm.LogNormal("beta", mu=2.5, sigma=1.0)

    rho = (1.0 - 2.0 * q[participant_id])[:, None]
    expo = pt.as_tensor_variable(np.array(WIDTHS, dtype="float64") - 2.0)[None, :]
    weights = retention ** expo
    weights = weights / pt.sum(weights, axis=1, keepdims=True)

    def score(side):
        mismatch = pt.sqr(imb[side] - expected_imbalance(effw[side], rho))
        return -pt.sum(weights * mismatch, axis=1)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score("a") - score("b"))))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
