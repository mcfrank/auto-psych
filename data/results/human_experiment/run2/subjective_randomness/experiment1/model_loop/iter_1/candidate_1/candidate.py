"""PyMC model for the Beta log-likelihood hypothesis.

People judge randomness by the log-likelihood that a sequence's alternation rate was drawn from a subjective Beta distribution skewed toward high alternations. Rather than using an ad-hoc piecewise distance function, this statistically grounded mechanism naturally provides a smooth, asymmetric cognitive penalty for both over- and under-alternating sequences, while maintaining a symmetric penalty for overall imbalance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    alpha = pm.HalfNormal("alpha", sigma=10.0)
    beta_param = pm.HalfNormal("beta_param", sigma=10.0)
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)

    beta_scale = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Clip p_alts to avoid log(0)
    p_a_safe = pt.clip(p_alts_a, 1e-4, 1.0 - 1e-4)
    p_b_safe = pt.clip(p_alts_b, 1e-4, 1.0 - 1e-4)

    # Beta logpdf (omitting the normalizing constant Beta(alpha, beta_param) since it cancels out in the difference)
    # log(x^(a-1) * (1-x)^(b-1)) = (a-1)*log(x) + (b-1)*log(1-x)
    logpdf_a = (alpha - 1.0) * pt.log(p_a_safe) + (beta_param - 1.0) * pt.log(
        1.0 - p_a_safe
    )
    logpdf_b = (alpha - 1.0) * pt.log(p_b_safe) + (beta_param - 1.0) * pt.log(
        1.0 - p_b_safe
    )

    score_a = logpdf_a - w_imb * imbalance_a
    score_b = logpdf_b - w_imb * imbalance_b

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta_scale * (score_a - score_b) + side_bias),
    )

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
