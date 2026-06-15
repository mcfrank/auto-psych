"""
People judge a sequence as more random-looking when its alternation rate is
close to an internal ideal, but penalize under-alternation (too predictable)
more harshly than over-alternation. This asymmetric V-shaped sensitivity around
the prototype alternation rate is the single mechanism driving randomness
judgments.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta_under = pm.HalfNormal("beta_under", sigma=5.0)
    beta_over = pm.HalfNormal("beta_over", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Asymmetric penalty: under-alternation (dev < 0) penalized at rate
    # beta_under, over-alternation (dev >= 0) penalized at rate beta_over.
    dev_a = p_alts_a - theta_alt
    dev_b = p_alts_b - theta_alt
    score_a = pt.switch(dev_a < 0, beta_under * dev_a, -beta_over * dev_a)
    score_b = pt.switch(dev_b < 0, beta_under * dev_b, -beta_over * dev_b)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b + side_bias))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
