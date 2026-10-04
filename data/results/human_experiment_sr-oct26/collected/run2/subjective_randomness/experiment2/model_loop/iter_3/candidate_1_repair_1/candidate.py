"""Two-flip-memory subjective coin.

People read a sequence flip by flip and, holding only the last two flips in
working memory, predict each next flip from a belief about how a random coin
behaves after a switch versus after a repeat. A sequence looks random to the
extent that its flips match these expectations (its probability under that
subjective coin), so both long streaks and strict alternation violate
expectation; people choose the sequence the subjective coin finds more
probable, apart from a small personal left/right lean.

The belief about the coin is shared (repair: a person-specific belief made the
fit's geometry unidentifiable); only the response lean is person-specific.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def counts(seq):
        seq = seq.strip().upper()
        trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
        ss = sr = rs = rr = 0
        for prev, cur in zip(trans, trans[1:]):
            if prev == 1 and cur == 1:
                ss += 1
            elif prev == 1:
                sr += 1
            elif cur == 1:
                rs += 1
            else:
                rr += 1
        first_s = float(trans[0]) if trans else 0.0
        first_r = 1.0 - first_s if trans else 0.0
        return np.array([ss, sr, rs, rr, first_s, first_r], dtype=float)

    d = counts(sequence_a) - counts(sequence_b)
    return {
        "d_ss": d[0],
        "d_sr": d[1],
        "d_rs": d[2],
        "d_rr": d[3],
        "d_first_s": d[4],
        "d_first_r": d[5],
    }


N_SLOTS = 400

with pm.Model() as model:
    d_ss = pm.Data("d_ss", np.zeros(1, dtype="float64"))
    d_sr = pm.Data("d_sr", np.zeros(1, dtype="float64"))
    d_rs = pm.Data("d_rs", np.zeros(1, dtype="float64"))
    d_rr = pm.Data("d_rr", np.zeros(1, dtype="float64"))
    d_first_s = pm.Data("d_first_s", np.zeros(1, dtype="float64"))
    d_first_r = pm.Data("d_first_r", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Shared belief (logit scale): switch probability after a switch (a)
    # and after a repeat (b).
    logit_a = pm.Normal("logit_a", mu=0.0, sigma=1.0)
    logit_b = pm.Normal("logit_b", mu=0.5, sigma=1.0)

    log_a = -pm.math.log1pexp(-logit_a)
    log_1a = -pm.math.log1pexp(logit_a)
    log_b = -pm.math.log1pexp(-logit_b)
    log_1b = -pm.math.log1pexp(logit_b)
    m = 0.5 * (pm.math.sigmoid(logit_a) + pm.math.sigmoid(logit_b))

    # How strongly the felt probability difference drives the choice.
    beta = pm.HalfNormal("beta", sigma=1.5)

    # Person-specific left/right response lean (non-centred).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = (sigma_side * z_side)[participant_id]

    dloglik = (
        d_ss * log_a + d_sr * log_1a + d_rs * log_b + d_rr * log_1b
        + d_first_s * pm.math.log(m) + d_first_r * pm.math.log(1.0 - m)
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * dloglik + side))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
