"""Gambler's fallacy with fading memory (law-of-small-numbers subjective coin).

People read a sequence flip by flip expecting the coin to balance out the
flips they have just seen, recent flips weighing more (exponentially fading
memory). A sequence looks random to the extent its flips match these balancing
expectations, i.e. by its log-probability under that self-correcting subjective
coin. Expectation of a switch builds up as a streak continues, so long runs are
penalised beyond their switch count. People differ in the strength of the
balancing expectation. Choice follows the ratio of the two sequences'
probabilities under the person's subjective coin (Luce rule), so the strength
of the expectation alone sets how decisive choices are.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

L = 7  # predicted positions (flips 2..8) and maximum lag


def _seq_arrays(seq):
    """prod[i-1, l-1] = s_i * s_{i-l} (s = +1 H, -1 T); mask[i-1] = flip i exists."""
    s = [1.0 if c == "H" else -1.0 for c in seq.strip().upper()]
    prod = np.zeros((L, L))
    mask = np.zeros(L)
    for i in range(1, len(s)):
        mask[i - 1] = 1.0
        for l in range(1, i + 1):
            prod[i - 1, l - 1] = s[i] * s[i - l]
    return prod, mask


def prepare_observed(rows):
    pa, ma, pb, mb = [], [], [], []
    for r in rows:
        x, m = _seq_arrays(r["sequence_a"])
        pa.append(x)
        ma.append(m)
        x, m = _seq_arrays(r["sequence_b"])
        pb.append(x)
        mb.append(m)
    return {
        "prod_a": np.asarray(pa, dtype="float64"),
        "mask_a": np.asarray(ma, dtype="float64"),
        "prod_b": np.asarray(pb, dtype="float64"),
        "mask_b": np.asarray(mb, dtype="float64"),
        "participant_id": np.asarray([int(r["participant_id"]) for r in rows], dtype="int64"),
        "chose_left": np.asarray([int(r.get("chose_left", 0)) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    prod_a = pm.Data("prod_a", np.zeros((1, L, L), dtype="float64"))
    mask_a = pm.Data("mask_a", np.zeros((1, L), dtype="float64"))
    prod_b = pm.Data("prod_b", np.zeros((1, L, L), dtype="float64"))
    mask_b = pm.Data("mask_b", np.zeros((1, L), dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific strength of the balancing (gambler's fallacy) expectation.
    g_mu = pm.Normal("g_mu", mu=0.5, sigma=1.5)
    g_sigma = pm.HalfNormal("g_sigma", sigma=1.0)
    g_z = pm.Normal("g_z", mu=0.0, sigma=1.0, shape=400)
    g = (g_mu + g_sigma * g_z)[participant_id]

    # Shared memory decay across lags.
    w = pm.Beta("w", alpha=2.0, beta=2.0)
    w_pow = w ** pt.arange(L)

    def log_prob(prod, mask):
        # drive_i = sum_l w^(l-1) s_i s_{i-l}: positive when flip i repeats
        # recent flips; a balancing expectation (g > 0) makes that improbable.
        drive = pt.sum(prod * w_pow[None, None, :], axis=2)
        return pt.sum(mask * pt.log(pm.math.sigmoid(-g[:, None] * drive)), axis=1)

    score_a = log_prob(prod_a, mask_a)
    score_b = log_prob(prod_b, mask_b)

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(score_a - score_b), 1e-6, 1 - 1e-6)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
