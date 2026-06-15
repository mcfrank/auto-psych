"""Continuous Bayesian Diagnosticity Cognitive Model.

Instead of a mixture of rigid point-mass alternatives (e.g., exactly 0.85 bias, 
0.95 alternating), this model uses continuous Beta priors. A single Beta(c, c) 
prior over bias unifies both "heads-biased" and "tails-biased" alternatives. 
A single Beta(c, c) prior over transition probabilities unifies both "streaky" 
and "alternating" alternatives. The concentration parameters c control how 
extreme the alternatives are expected to be.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

def _continuous_bias_log_prob(n, h, conc):
    """Log marginal likelihood of h heads in n flips under p ~ Beta(conc, conc)."""
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    t_f = n_f - h_f
    
    # log B(h+conc, t+conc) - log B(conc, conc)
    log_B_post = pt.gammaln(h_f + conc) + pt.gammaln(t_f + conc) - pt.gammaln(n_f + 2.0 * conc)
    log_B_prior = 2.0 * pt.gammaln(conc) - pt.gammaln(2.0 * conc)
    
    log_prob = log_B_post - log_B_prior
    
    # Length normalization for mixed-length sequences
    denom = pt.maximum(n_f, 1.0)
    return log_prob / denom

def _continuous_markov_log_prob(n, alts, conc):
    """Log marginal likelihood of transitions under p_switch ~ Beta(conc, conc)."""
    n_f = pt.cast(n, "float64")
    alts_f = pt.cast(alts, "float64")
    stays_f = pt.maximum(n_f - 1.0, 0.0) - alts_f
    n_trans = pt.maximum(n_f - 1.0, 0.0)
    
    log_B_post = pt.gammaln(alts_f + conc) + pt.gammaln(stays_f + conc) - pt.gammaln(n_trans + 2.0 * conc)
    log_B_prior = 2.0 * pt.gammaln(conc) - pt.gammaln(2.0 * conc)
    
    # The first flip is always log(0.5).
    log_prob = np.log(0.5) + (log_B_post - log_B_prior)
    
    denom = pt.maximum(n_f, 1.0)
    return log_prob / denom

def _fair_log_prob(n):
    n_f = pt.cast(n, "float64")
    log_prob = n_f * np.log(0.5)
    denom = pt.maximum(n_f, 1.0)
    return log_prob / denom

def _score_sequence(n, h, alts, alt_prior, c_bias, c_markov):
    fair = _fair_log_prob(n)
    bias_alt = _continuous_bias_log_prob(n, h, c_bias)
    markov_alt = _continuous_markov_log_prob(n, alts, c_markov)
    
    # logsumexp over alternative models
    alternatives = _logsumexp2(
        pt.log(alt_prior) + markov_alt,
        pt.log(1.0 - alt_prior) + bias_alt
    )
    
    # Score is the log likelihood ratio of Fair vs Alternatives
    return fair - alternatives

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    # Cognitive parameters
    # Prior probability of the Markov alternative vs the Bias alternative
    alt_prior = pm.Uniform("alt_prior", lower=0.01, upper=0.99)
    
    # Concentration parameters for the Beta(c, c) priors.
    # Values < 1 imply U-shaped priors (expecting extreme biases or extreme dependencies).
    # Values > 1 imply unimodal priors centered at 0.5.
    c_bias = pm.Uniform("c_bias", lower=0.01, upper=20.0)
    c_markov = pm.Uniform("c_markov", lower=0.01, upper=20.0)
    
    # Decision parameters
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Score each sequence
    score_a = _score_sequence(n_a, h_a, alts_a, alt_prior, c_bias, c_markov)
    score_b = _score_sequence(n_b, h_b, alts_b, alt_prior, c_bias, c_markov)

    # Deterministic probability of choosing left
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
