"""People evaluate sequences using the same two-dimensional prototype (alternation
rate + balance) as the Gaussian prototype account, but their sensitivity to
deviations scales linearly with sequence length — a longer sequence that deviates
from the prototype provides proportionally stronger evidence of non-randomness
than a shorter sequence with the same deviation proportion."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.65)
    beta_alt = pm.HalfNormal("beta_alt", sigma=5.0)
    beta_bal = pm.HalfNormal("beta_bal", sigma=5.0)

    # Normalize by minimum sequence length (4) so beta_alt/beta_bal are comparable
    # to the unweighted prototype model; the n/4 factor adds length-dependent scaling
    n_a_f = pt.cast(n_a, "float64") / 4.0
    n_b_f = pt.cast(n_b, "float64") / 4.0

    score_a = -n_a_f * (beta_alt * (p_alts_a - theta_alt) ** 2 + beta_bal * (p_a - 0.5) ** 2)
    score_b = -n_b_f * (beta_alt * (p_alts_b - theta_alt) ** 2 + beta_bal * (p_b - 0.5) ** 2)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
