"""People judge a sequence by its single most striking copyable stretch: the
longest contiguous stretch that keeps repeating one short unit of 1-4 flips
(streak, strict alternation, or a repeating motif such as HHTHHT). The number of
flips in that stretch predictable by copying the unit (stretch length minus the
unit length) alone decides how patterned the sequence looks; people pick the
sequence whose most copyable stretch is shorter, with person-specific strength
and a personal left/right lean.
"""
import numpy as np
import pymc as pm

N_SLOTS = 400


def _max_copy_excess(seq):
    s = seq.strip().upper()
    n = len(s)
    best = 0
    for p in range(1, 5):
        if p >= n:
            break
        # longest run of consecutive positions i with s[i] == s[i+p]
        run = 0
        for i in range(n - p):
            if s[i] == s[i + p]:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return float(best)


def compute_features(sequence_a, sequence_b):
    n = len(sequence_a.strip())
    return {
        "copy_a": _max_copy_excess(sequence_a) / n,
        "copy_b": _max_copy_excess(sequence_b) / n,
    }


with pm.Model() as model:
    copy_a = pm.Data("copy_a", np.zeros(1, dtype="float64"))
    copy_b = pm.Data("copy_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_beta = pm.Normal("mu_beta", mu=2.0, sigma=2.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=N_SLOTS)
    beta = mu_beta + sigma_beta * z_beta

    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", 0.0, 1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    eta = side[participant_id] + beta[participant_id] * (copy_b - copy_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
