import numpy as np
import pymc as pm
import pytensor.tensor as pt

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

def _continuous_bias_log_prob(n, h, conc):
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    t_f = n_f - h_f
    
    log_B_post = pt.gammaln(h_f + conc) + pt.gammaln(t_f + conc) - pt.gammaln(n_f + 2.0 * conc)
    log_B_prior = 2.0 * pt.gammaln(conc) - pt.gammaln(2.0 * conc)
    
    log_prob = log_B_post - log_B_prior
    denom = pt.maximum(n_f, 1.0)
    return log_prob / denom

def _continuous_markov_log_prob(n, alts, conc):
    n_f = pt.cast(n, "float64")
    alts_f = pt.cast(alts, "float64")
    stays_f = pt.maximum(n_f - 1.0, 0.0) - alts_f
    n_trans = pt.maximum(n_f - 1.0, 0.0)
    
    log_B_post = pt.gammaln(alts_f + conc) + pt.gammaln(stays_f + conc) - pt.gammaln(n_trans + 2.0 * conc)
    log_B_prior = 2.0 * pt.gammaln(conc) - pt.gammaln(2.0 * conc)
    
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
    
    alternatives = _logsumexp2(
        pt.log(alt_prior) + markov_alt,
        pt.log(1.0 - alt_prior) + bias_alt
    )
    return fair - alternatives

with pm.Model() as m:
    n = pm.Data("n", np.array([10]))
    h = pm.Data("h", np.array([3]))
    alts = pm.Data("alts", np.array([2]))
    
    c_bias = pm.Uniform("c_bias", 0.01, 10.0)
    c_markov = pm.Uniform("c_markov", 0.01, 10.0)
    alt_prior = pm.Uniform("alt_prior", 0.01, 0.99)
    
    score = _score_sequence(n, h, alts, alt_prior, c_bias, c_markov)

print("Compiles fine!")
