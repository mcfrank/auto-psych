import pymc as pm
import pytensor.tensor as pt
import numpy as np

with pm.Model() as model:
    n_a = pm.Data("n_a", np.array([10], dtype="int64"))
    h_a = pm.Data("h_a", np.array([5], dtype="int64"))
    alts_a = pm.Data("alts_a", np.array([5], dtype="int64"))

    n_b = pm.Data("n_b", np.array([10], dtype="int64"))
    h_b = pm.Data("h_b", np.array([5], dtype="int64"))
    alts_b = pm.Data("alts_b", np.array([5], dtype="int64"))

    ideal_alts = pm.Beta("ideal_alts", alpha=5.0, beta=5.0)
    w_alts = pm.HalfNormal("w_alts", sigma=2.0)
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Use pt.clip to avoid zero or negative n_alts
    n_alts_a = pt.clip(n_a - 1, 1, 1000)
    n_alts_b = pt.clip(n_b - 1, 1, 1000)

    log_p_h_a = pm.logp(pm.Binomial.dist(n=n_a, p=0.5), h_a)
    log_p_alts_a = pm.logp(pm.Binomial.dist(n=n_alts_a, p=ideal_alts), alts_a)
    score_a = log_p_h_a + w_alts * log_p_alts_a

    log_p_h_b = pm.logp(pm.Binomial.dist(n=n_b, p=0.5), h_b)
    log_p_alts_b = pm.logp(pm.Binomial.dist(n=n_alts_b, p=ideal_alts), alts_b)
    score_b = log_p_h_b + w_alts * log_p_alts_b

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.array([0], dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)

print("Graph built successfully!")
