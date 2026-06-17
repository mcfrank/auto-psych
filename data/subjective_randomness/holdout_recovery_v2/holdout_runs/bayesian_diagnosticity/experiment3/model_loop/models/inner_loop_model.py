"""
People judge a sequence as more random-looking when it is close to an internal
2D prototype specifying both an ideal alternation rate and balanced heads and
tails. Closeness follows Gaussian decay in both dimensions — squared deviations
in alternation rate and in balance each reduce the randomness impression, and
the two dimensions contribute independently by adding their squared distances.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1))
    p_a = pm.Data("p_a", np.zeros(1))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1))
    p_b = pm.Data("p_b", np.zeros(1))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.65)
    beta_alt = pm.HalfNormal("beta_alt", sigma=5.0)
    beta_bal = pm.HalfNormal("beta_bal", sigma=5.0)

    score_a = -beta_alt * (p_alts_a - theta_alt) ** 2 - beta_bal * (p_a - 0.5) ** 2
    score_b = -beta_alt * (p_alts_b - theta_alt) ** 2 - beta_bal * (p_b - 0.5) ** 2

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
