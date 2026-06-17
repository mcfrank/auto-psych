import numpy as np
import pymc as pm
import pytensor.tensor as pt

def _normalized_iid_log_prob(n, h, p_heads):
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    tails_f = n_f - h_f
    denom = pt.maximum(n_f, 1.0)
    return (h_f * np.log(p_heads) + tails_f * np.log(1.0 - p_heads)) / denom

def _normalized_markov_log_prob(n, alts, switch_prob):
    n_f = pt.cast(n, "float64")
    alts_f = pt.cast(alts, "float64")
    stays_f = pt.maximum(n_f - 1.0, 0.0) - alts_f
    denom = pt.maximum(n_f, 1.0)
    return (
        np.log(0.5) + alts_f * pt.log(switch_prob) + stays_f * pt.log(1.0 - switch_prob)
    ) / denom

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)
    alt_switch_prob = pm.Uniform("alt_switch_prob", lower=0.51, upper=0.99)
    
    fair_a = _normalized_iid_log_prob(n_a, h_a, 0.5)
    alt_a = _normalized_markov_log_prob(n_a, alts_a, alt_switch_prob)
    score_a = fair_a - alt_a
    
    fair_b = _normalized_iid_log_prob(n_b, h_b, 0.5)
    alt_b = _normalized_markov_log_prob(n_b, alts_b, alt_switch_prob)
    score_b = fair_b - alt_b
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
