"""Two-flip-memory subjective coin with Luce choice on its probabilities.

People judge randomness with a subjective coin that remembers only its last two
flips: the probability that the next flip switches sides depends on whether the
previous transition was a repeat or a switch. A sequence looks random to the
extent it is probable under this coin, and people choose between the two
sequences in proportion to those probabilities (Luce choice, no separate
decisiveness parameter), apart from a small personal left/right lean.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    """Differences (left minus right) in transition-pattern counts."""

    def counts(seq):
        seq = seq.strip().upper()
        trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]  # 1 = switch
        first_s = float(trans[0]) if trans else 0.0
        first_r = 1.0 - first_s if trans else 0.0
        rs = rr = ss = sr = 0.0
        for prev, cur in zip(trans, trans[1:]):
            if prev == 0 and cur == 1:
                rs += 1
            elif prev == 0 and cur == 0:
                rr += 1
            elif prev == 1 and cur == 1:
                ss += 1
            else:
                sr += 1
        return first_s, first_r, rs, rr, ss, sr

    ca, cb = counts(sequence_a), counts(sequence_b)
    names = ["d_first_s", "d_first_r", "d_rs", "d_rr", "d_ss", "d_sr"]
    return {n: float(x - y) for n, x, y in zip(names, ca, cb)}


with pm.Model() as model:
    d_first_s = pm.Data("d_first_s", np.zeros(1, dtype="float64"))
    d_first_r = pm.Data("d_first_r", np.zeros(1, dtype="float64"))
    d_rs = pm.Data("d_rs", np.zeros(1, dtype="float64"))
    d_rr = pm.Data("d_rr", np.zeros(1, dtype="float64"))
    d_ss = pm.Data("d_ss", np.zeros(1, dtype="float64"))
    d_sr = pm.Data("d_sr", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Subjective switch probabilities (logit scale): first transition,
    # after a repeat, after a switch.
    logit_c = pm.Normal("logit_c", 0.0, 1.5)
    logit_a = pm.Normal("logit_a", 0.0, 1.5)
    logit_b = pm.Normal("logit_b", 0.0, 1.5)

    def log_sig(x):
        return -pm.math.log1pexp(-x)

    # Log probability of the left sequence minus that of the right one.
    lp_diff = (
        d_first_s * log_sig(logit_c) + d_first_r * log_sig(-logit_c)
        + d_rs * log_sig(logit_a) + d_rr * log_sig(-logit_a)
        + d_ss * log_sig(logit_b) + d_sr * log_sig(-logit_b)
    )

    # Personal left/right lean (non-centred), slots well beyond participants so far.
    lean_mu = pm.Normal("lean_mu", 0.0, 0.5)
    lean_sigma = pm.HalfNormal("lean_sigma", 0.3)
    lean_z = pm.Normal("lean_z", 0.0, 1.0, shape=400)
    lean = lean_mu + lean_sigma * lean_z

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(lp_diff + lean[participant_id]))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
