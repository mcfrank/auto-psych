"""Context learner surprise.

People read a sequence flip by flip and, with a short memory of the last two
flips, keep trying to predict the next flip from what followed the same
two-flip context earlier in the sequence (an online pattern learner); a
sequence looks random to the extent that these running predictions keep
failing, i.e. by the average surprise the learner experiences. Long streaks
and repeating patterns become predictable to such a learner and so look
non-random. People differ in how strongly this felt surprise drives choice.

The learner's predictive probability for the flip at position i is
(n_same + a) / (n_total + 2a): n_total earlier occurrences of the current
context (the previous min(2, i) flips), n_same of which were followed by the
flip that actually came; a is the learner's prior pseudo-count (how slowly it
picks up patterns), a free parameter.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_POS = 7  # predicted positions 1..7 for sequences of length <= 8
ORDER = 2
N_SLOTS = 400


def _learner_counts(seq):
    seq = seq.strip().upper()
    same = np.zeros(MAX_POS)
    total = np.zeros(MAX_POS)
    mask = np.zeros(MAX_POS)
    for i in range(1, len(seq)):
        k = min(ORDER, i)
        ctx = seq[i - k:i]
        n_same = 0
        n_total = 0
        for j in range(k, i):
            if seq[j - k:j] == ctx:
                n_total += 1
                if seq[j] == seq[i]:
                    n_same += 1
        same[i - 1] = n_same
        total[i - 1] = n_total
        mask[i - 1] = 1.0
    return same, total, mask


def compute_features(sequence_a, sequence_b):
    out = {}
    for side, seq in (("a", sequence_a), ("b", sequence_b)):
        same, total, mask = _learner_counts(seq)
        for j in range(MAX_POS):
            out[f"cl_same_{side}{j}"] = float(same[j])
            out[f"cl_total_{side}{j}"] = float(total[j])
            out[f"cl_mask_{side}{j}"] = float(mask[j])
    return out


def _stack(side, kind):
    return pt.stack(
        [pm.Data(f"cl_{kind}_{side}{j}", np.zeros(1, dtype="float64")) for j in range(MAX_POS)],
        axis=1,
    )


with pm.Model() as model:
    same_a, total_a, mask_a = _stack("a", "same"), _stack("a", "total"), _stack("a", "mask")
    same_b, total_b, mask_b = _stack("b", "same"), _stack("b", "total"), _stack("b", "mask")
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Learner's prior pseudo-count: small = picks up patterns fast.
    a = pm.LogNormal("a", mu=0.0, sigma=1.0)

    def mean_surprise(same, total, mask):
        p = (same + a) / (total + 2.0 * a)
        s = -pt.log2(p) * mask
        return pt.sum(s, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)

    r_a = mean_surprise(same_a, total_a, mask_a)
    r_b = mean_surprise(same_b, total_b, mask_b)

    # Person-specific sensitivity to the felt surprise (non-centred).
    mu_beta = pm.Normal("mu_beta", mu=2.0, sigma=3.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=3.0)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = mu_beta + sigma_beta * z_beta

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * (r_a - r_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
