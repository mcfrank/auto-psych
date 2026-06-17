"""
People evaluate sequences against a 2D prototype (ideal alternation rate + balanced outcomes),
but their sensitivity in the alternation dimension is asymmetric: sequences that are too streaky
(alternation below the ideal) are penalized more heavily than sequences that are too alternating,
because streaks feel more distinctively non-random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Prototype: ideal alternation rate
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Asymmetric sensitivity: separate quadratic slopes for streaky vs over-alternating
    beta_low = pm.HalfNormal("beta_low", sigma=5.0)   # penalty when alt < theta (streaky)
    beta_high = pm.HalfNormal("beta_high", sigma=5.0)  # penalty when alt > theta (over-alternating)

    # Symmetric sensitivity to outcome imbalance
    beta_bal = pm.HalfNormal("beta_bal", sigma=5.0)

    # Side bias
    side_bias = pm.Normal("side_bias", mu=0, sigma=1)

    # Piecewise quadratic alternation penalty
    dev_a = p_alts_a - theta_alt
    dev_b = p_alts_b - theta_alt
    beta_a = pt.switch(pt.lt(dev_a, 0), beta_low, beta_high)
    beta_b = pt.switch(pt.lt(dev_b, 0), beta_low, beta_high)
    alt_score_a = -beta_a * dev_a ** 2
    alt_score_b = -beta_b * dev_b ** 2

    # Symmetric balance penalty
    bal_score_a = -beta_bal * imbalance_a ** 2
    bal_score_b = -beta_bal * imbalance_b ** 2

    score_diff = (alt_score_a + bal_score_a) - (alt_score_b + bal_score_b) + side_bias
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
