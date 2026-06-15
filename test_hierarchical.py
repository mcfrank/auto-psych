import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALT_SWITCH_PROB = 0.95
STREAK_SWITCH_PROB = 0.15
BIAS_HEAD_PROB = 0.85

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

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
        np.log(0.5) + alts_f * np.log(switch_prob) + stays_f * np.log(1.0 - switch_prob)
    ) / denom

def _score_sequence(n, h, alts, alt_prior, bias_share):
    fair = _normalized_iid_log_prob(n, h, 0.5)
    alternating = _normalized_markov_log_prob(n, alts, ALT_SWITCH_PROB)
    streaky = _normalized_markov_log_prob(n, alts, STREAK_SWITCH_PROB)
    bias_h = _normalized_iid_log_prob(n, h, BIAS_HEAD_PROB)
    bias_t = _normalized_iid_log_prob(n, h, 1.0 - BIAS_HEAD_PROB)
    biased = _logsumexp2(np.log(0.5) + bias_h, np.log(0.5) + bias_t)

    alt_weight = alt_prior
    bias_weight = (1.0 - alt_prior) * bias_share
    streak_weight = (1.0 - alt_prior) * (1.0 - bias_share)
    
    alternatives = _logsumexp2(
        _logsumexp2(
            pt.log(alt_weight) + alternating,
            pt.log(bias_weight) + biased,
        ),
        pt.log(streak_weight) + streaky,
    )
    return fair - alternatives

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    
    # Cognitive parameters
    alt_prior = pm.Uniform("alt_prior", lower=0.01, upper=0.98)
    bias_share = pm.Uniform("bias_share", lower=0.01, upper=0.99)
    
    # Hierarchical Beta (decision noise)
    beta_mu = pm.Normal("beta_mu", mu=1.0, sigma=2.0)
    beta_sigma = pm.HalfNormal("beta_sigma", sigma=2.0)
    beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=100)
    beta = pt.exp(beta_mu + beta_sigma * beta_offset)
    
    # Hierarchical Side Bias
    side_bias_mu = pm.Normal("side_bias_mu", mu=0.0, sigma=1.0)
    side_bias_sigma = pm.HalfNormal("side_bias_sigma", sigma=1.0)
    side_bias_offset = pm.Normal("side_bias_offset", mu=0.0, sigma=1.0, shape=100)
    side_bias = side_bias_mu + side_bias_sigma * side_bias_offset

    # Score each sequence
    score_a = _score_sequence(n_a, h_a, alts_a, alt_prior, bias_share)
    score_b = _score_sequence(n_b, h_b, alts_b, alt_prior, bias_share)

    p_beta = beta[participant_id]
    p_side_bias = side_bias[participant_id]

    # Deterministic probability of choosing left
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(p_beta * (score_a - score_b) + p_side_bias)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)

from pathlib import Path
from src.models.pymc_inference import load_pymc_model, observed_response_data
import sys
# fake module
sys.modules['candidate'] = sys.modules[__name__]
print('observed:', observed_response_data(model))
