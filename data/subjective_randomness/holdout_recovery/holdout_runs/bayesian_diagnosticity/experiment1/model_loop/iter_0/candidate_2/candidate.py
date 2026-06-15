import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs for alternation rates
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: sensitivity to alternation rate differences
    # Normal prior centered at 0 to allow preference for either higher or lower alternation rates
    beta_alts = pm.Normal("beta_alts", mu=0.0, sigma=5.0)

    # Heuristic: Difference in proportion of alternations
    log_odds = beta_alts * (p_alts_a - p_alts_b)
    
    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(log_odds))

    # Observed response: the pm.Data tensor is passed directly to observed=
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
