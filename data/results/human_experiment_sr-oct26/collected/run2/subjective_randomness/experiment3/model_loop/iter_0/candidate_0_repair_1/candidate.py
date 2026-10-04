"""Indifference band in the decision rule over a simple, shared tally/switch impression.

People do not choose in proportion to how much more random one sequence looks:
when the felt difference between the two sequences is small they are
indifferent and pick near a coin flip; only the part of the difference that
exceeds a shared indifference band pushes them toward a side. The evidence is
deliberately simple and shared by everyone: closeness of the running
H-minus-T tally span to an expected span, and closeness of the switching rate
to an ideal rate. The band is smooth (d - delta * tanh(d / delta)): flat near
zero, a shift by delta far from it.
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


with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))

    # Evidence: closeness of tally span to an expected span (share of length).
    # (s_b - c)^2 - (s_a - c)^2 = (s_b^2 - s_a^2) - 2c (s_b - s_a): written with
    # two free coefficients so the expected span and its weight do not trade off.
    a_span_sq = pm.Normal("a_span_sq", mu=0.0, sigma=3.0)
    a_span = pm.Normal("a_span", mu=0.0, sigma=3.0)
    # Closeness of the switching rate to an ideal rate.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    beta_sw = pm.HalfNormal("beta_sw", sigma=10.0)

    evidence = (
        a_span_sq * (span_b**2 - span_a**2)
        + a_span * (span_b - span_a)
        + beta_sw * ((sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2)
    )

    # The hypothesis: a shared, smooth indifference band (half-width delta, logit units).
    delta = pm.LogNormal("delta", mu=-1.0, sigma=0.7)
    banded = evidence - delta * pt.tanh(evidence / delta)

    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(banded), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
