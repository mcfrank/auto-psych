"""
Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence for randomness. This accumulated evidence is then discounted by the sequence's deviation from a messy prototype—an ideal positive imbalance and an ideal alternation rate—using an asymmetric quadratic penalty that punishes under-alternation (streaks) with a different, flexibly stronger weight than over-alternation.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    w_baseline = pm.HalfNormal("w_baseline", sigma=2.0)
    ideal_imb = pm.Uniform("ideal_imb", lower=0.0, upper=0.5)
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)
    
    ideal_alt = pm.Uniform("ideal_alt", lower=0.0, upper=1.0)
    w_under = pm.HalfNormal("w_under", sigma=10.0)
    w_over = pm.HalfNormal("w_over", sigma=10.0)

    tau = pm.HalfNormal("tau", sigma=1.0)

    # Calculate evidence for A
    diff_alt_a = p_alts_a - ideal_alt
    penalty_alt_a = pt.switch(diff_alt_a < 0, w_under * (diff_alt_a ** 2), w_over * (diff_alt_a ** 2))
    penalty_imb_a = w_imb * ((imbalance_a - ideal_imb) ** 2)
    ev_a = w_baseline * n_a - penalty_imb_a - penalty_alt_a

    # Calculate evidence for B
    diff_alt_b = p_alts_b - ideal_alt
    penalty_alt_b = pt.switch(diff_alt_b < 0, w_under * (diff_alt_b ** 2), w_over * (diff_alt_b ** 2))
    penalty_imb_b = w_imb * ((imbalance_b - ideal_imb) ** 2)
    ev_b = w_baseline * n_b - penalty_imb_b - penalty_alt_b

    # Choice probability
    # We use a softmax/sigmoid mapping: P(choose A) = sigmoid(tau * (ev_a - ev_b))
    # We clamp it for numerical safety as requested.
    p_left_raw = pm.math.sigmoid(tau * (ev_a - ev_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
