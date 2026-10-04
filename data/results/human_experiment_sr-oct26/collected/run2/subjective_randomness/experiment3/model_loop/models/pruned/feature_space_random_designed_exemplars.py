"""Exemplar similarity in a perceptual feature space (random vs designed).

People represent a sequence by its switch rate, its running heads-minus-tails
tally span (share of the length) and whether it visibly repeats a unit, and
compare it with remembered exemplars: one "typical random" sequence (whose
tally span is person-specific) and designed sequences of the same length (a
streak, strict alternation, and the repeating units HHT, HHTT, HHHT). A
sequence looks random to the extent that its Gaussian similarity to the random
exemplar exceeds its summed similarity to the designed exemplars; people pick
the sequence with the higher log similarity ratio (a small
left/right lean).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

DESIGNED_UNITS = ["H", "HT", "HHT", "HHTT", "HHHT"]


def _switch_rate(seq):
    return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)


def _span_share(seq):
    tally, hi, lo = 0, 0, 0
    for c in seq:
        tally += 1 if c == "H" else -1
        hi = max(hi, tally)
        lo = min(lo, tally)
    return (hi - lo) / len(seq)


def _periodic(seq):
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if all(seq[i] == seq[i + p] for i in range(n - p)):
            return 1.0
    return 0.0


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    n = len(a)
    out = {
        "s_a": _switch_rate(a), "r_a": _span_share(a), "per_a": _periodic(a),
        "s_b": _switch_rate(b), "r_b": _span_share(b), "per_b": _periodic(b),
    }
    for j, unit in enumerate(DESIGNED_UNITS):
        ex = (unit * n)[:n]
        out[f"sD{j}"] = _switch_rate(ex)
        out[f"rD{j}"] = _span_share(ex)
    return out


N_SLOTS = 400
J = len(DESIGNED_UNITS)

with pm.Model() as model:
    s_a = pm.Data("s_a", np.zeros(1, dtype="float64"))
    r_a = pm.Data("r_a", np.zeros(1, dtype="float64"))
    per_a = pm.Data("per_a", np.zeros(1, dtype="float64"))
    s_b = pm.Data("s_b", np.zeros(1, dtype="float64"))
    r_b = pm.Data("r_b", np.zeros(1, dtype="float64"))
    per_b = pm.Data("per_b", np.zeros(1, dtype="float64"))
    sD = pt.stack([pm.Data(f"sD{j}", np.zeros(1, dtype="float64")) for j in range(J)], axis=1)
    rD = pt.stack([pm.Data(f"rD{j}", np.zeros(1, dtype="float64")) for j in range(J)], axis=1)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # The remembered random exemplar: shared switch rate, person-specific span.
    s_rand = pm.Beta("s_rand", alpha=6.0, beta=4.0)
    mu_r = pm.Normal("mu_r", mu=-0.5, sigma=1.0)
    sigma_r = pm.HalfNormal("sigma_r", sigma=1.0)
    z_r = pm.Normal("z_r", mu=0.0, sigma=1.0, shape=N_SLOTS)
    r_rand = pm.math.sigmoid(mu_r + sigma_r * z_r)[participant_id]

    # Attention-weighted squared distance (weights absorb the similarity scale).
    w_s = pm.LogNormal("w_s", mu=1.5, sigma=1.0)
    w_r = pm.LogNormal("w_r", mu=1.5, sigma=1.0)
    w_p = pm.LogNormal("w_p", mu=0.0, sigma=1.0)

    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = (sigma_side * z_side)[participant_id]

    def evidence(s, r, per):
        log_sim_rand = -(w_s * (s - s_rand) ** 2 + w_r * (r - r_rand) ** 2 + w_p * per)
        log_sim_des = -(
            w_s * (s[:, None] - sD) ** 2
            + w_r * (r[:, None] - rD) ** 2
            + w_p * (1.0 - per)[:, None]
        )
        return log_sim_rand - pt.logsumexp(log_sim_des, axis=1)

    logit = (evidence(s_a, r_a, per_a) - evidence(s_b, r_b, per_b)) + side
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
