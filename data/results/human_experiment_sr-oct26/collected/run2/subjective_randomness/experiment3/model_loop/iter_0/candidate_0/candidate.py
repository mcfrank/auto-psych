"""Indifference band in the decision rule over a tally-span / switch-ideal impression.

People do not choose in proportion to how much more random one sequence looks:
when the felt difference between the two sequences is small they are
indifferent and pick near a coin flip; only the part of the difference that
exceeds a shared indifference band pushes them toward a side. The evidence is
the simple impression of `tally_span_switch_rate_ideal_2` (running H-minus-T
tally span against a personal expected span, closeness of the switching rate
to a shared ideal, personal left/right lean). The band is smooth
(d - delta * tanh(d / delta)): flat near zero, a shift by delta far from it.
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
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Evidence: person-specific expected tally span (share of the length).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)

    # Person-specific sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Switching judged by closeness to a shared ideal rate, person-specific weight.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = mu_kappa + sigma_kappa * z_kappa

    # Person-specific left/right lean.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    sw_dist = (sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2
    evidence = sens * (dist_b - dist_a) + 4.0 * kappa[participant_id] * sw_dist

    # The hypothesis: a shared, smooth indifference band (half-width delta,
    # logit units) in the decision rule.
    log_delta = pm.Normal("log_delta", mu=-0.5, sigma=1.0)
    delta = pm.Deterministic("delta", pt.exp(log_delta))
    banded = evidence - delta * pt.tanh(evidence / delta)

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(banded + side[participant_id]), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
