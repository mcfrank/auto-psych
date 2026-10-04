"""Heavy-tailed (Cauchy) decision noise over a simple, shared tally/switch impression.

People's impression of which sequence is more random is corrupted by
heavy-tailed noise: usually small, occasionally a gross misreading of the pair.
The choice probability therefore follows a Cauchy CDF of the felt difference
(0.5 + atan(d)/pi) instead of a logistic: near-ties are decided sharply, but
clear-cut pairs approach consensus only slowly. The evidence is simple and
shared: closeness of the running H-minus-T tally span to an expected span, and
closeness of the switching rate to an ideal rate, plus a personal left/right lean.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a, sequence_b):
    def span_share(seq):
        tally, hi, lo = 0, 0, 0
        for c in seq:
            tally += 1 if c == "H" else -1
            hi = max(hi, tally)
            lo = min(lo, tally)
        return (hi - lo) / len(seq)

    def switch_rate(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Closeness of tally span to an expected span, written with two free
    # coefficients so the expected span and its weight do not trade off.
    a_span_sq = pm.Normal("a_span_sq", mu=0.0, sigma=3.0)
    a_span = pm.Normal("a_span", mu=0.0, sigma=3.0)
    # Closeness of the switching rate to a shared ideal rate.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    beta_sw = pm.HalfNormal("beta_sw", sigma=10.0)

    # Personal left/right lean (non-centred).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    evidence = (
        a_span_sq * (span_b**2 - span_a**2)
        + a_span * (span_b - span_a)
        + beta_sw * ((sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2)
        + side[participant_id]
    )

    # The hypothesis: heavy-tailed (Cauchy) decision noise.
    p = 0.5 + pt.arctan(evidence) / np.pi
    p_left = pm.Deterministic("p_left", pt.clip(p, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
