"""People judge randomness by typicality under their own imagined chance process
alone (no rival regular explanation): each person holds a personal second-order
belief about a fair coin -- how likely it switches right after a repeat and how
likely right after a switch -- and a sequence looks random to the extent this
imagined coin would readily produce it (per-flip log probability). The sequence
more typical of the person's imagined coin is chosen as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _counts(seq):
    s = seq.strip().upper()
    trans = [1 if x != y else 0 for x, y in zip(s, s[1:])]  # 1 = switch
    first_sw = float(trans[0]) if trans else 0.0
    sw_r = rep_r = sw_s = rep_s = 0.0
    for prev, cur in zip(trans, trans[1:]):
        if prev == 0:
            if cur:
                sw_r += 1
            else:
                rep_r += 1
        else:
            if cur:
                sw_s += 1
            else:
                rep_s += 1
    return first_sw, sw_r, rep_r, sw_s, rep_s


def compute_features(sequence_a, sequence_b):
    ca = _counts(sequence_a)
    cb = _counts(sequence_b)
    names = ["d_first_sw", "d_sw_after_rep", "d_rep_after_rep", "d_sw_after_sw", "d_rep_after_sw"]
    out = {k: float(a - b) for k, a, b in zip(names, ca, cb)}
    out["seq_len"] = float(len(sequence_a.strip()))
    return out


def _logsig(x):
    return -pt.softplus(-x)


with pm.Model() as model:
    d_first = pm.Data("d_first_sw", np.zeros(1, dtype="float64"))
    d_swr = pm.Data("d_sw_after_rep", np.zeros(1, dtype="float64"))
    d_repr = pm.Data("d_rep_after_rep", np.zeros(1, dtype="float64"))
    d_sws = pm.Data("d_sw_after_sw", np.zeros(1, dtype="float64"))
    d_reps = pm.Data("d_rep_after_sw", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.ones(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific beliefs (logit switch probability) after a repeat / after a switch.
    mu_r = pm.Normal("mu_r", mu=0.5, sigma=1.0)
    mu_s = pm.Normal("mu_s", mu=0.0, sigma=1.0)
    sigma_r = pm.HalfNormal("sigma_r", sigma=1.0)
    sigma_s = pm.HalfNormal("sigma_s", sigma=1.0)
    z_r = pm.Normal("z_r", 0.0, 1.0, shape=400)
    z_s = pm.Normal("z_s", 0.0, 1.0, shape=400)
    a_r = (mu_r + sigma_r * z_r)[participant_id]
    a_s = (mu_s + sigma_s * z_s)[participant_id]
    # First transition: the average of the person's two contexts.
    a_0 = 0.5 * (a_r + a_s)

    # Log-likelihood difference (a minus b) under the imagined coin.
    ll_diff = (
        d_first * _logsig(a_0)
        + d_swr * _logsig(a_r)
        + d_repr * _logsig(-a_r)
        + d_sws * _logsig(a_s)
        + d_reps * _logsig(-a_s)
    )
    # The first-transition count difference is 0 or +-1 and its term is included above;
    # the evidence is weighed per flip by a fitted power of length.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)
    score = beta * ll_diff / pt.power(seq_len, gamma)

    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(score), 1e-6, 1 - 1e-6))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
