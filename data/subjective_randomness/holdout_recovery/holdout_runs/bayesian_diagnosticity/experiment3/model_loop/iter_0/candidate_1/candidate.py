import numpy as np
import pymc as pm
import pytensor.tensor as pt

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

def _scaled_fair_log_prob(n, gamma):
    n_f = pt.cast(n, "float64")
    return (n_f ** (1.0 - gamma)) * np.log(0.5)

def _scaled_betabinomial_log_prob(n, k, alpha, beta_param, gamma):
    n_f = pt.cast(n, "float64")
    k_f = pt.cast(k, "float64")
    
    log_marg_lik = (
        pt.gammaln(k_f + alpha) 
        + pt.gammaln(n_f - k_f + beta_param) 
        - pt.gammaln(n_f + alpha + beta_param)
        + pt.gammaln(alpha + beta_param)
        - pt.gammaln(alpha)
        - pt.gammaln(beta_param)
    )
    
    denom = pt.maximum(n_f ** gamma, 1.0)
    return log_marg_lik / denom

def _scaled_markov_betabinomial_log_prob(n, alts, alpha, beta_param, gamma):
    n_f = pt.cast(n, "float64")
    alts_f = pt.cast(alts, "float64")
    stays_f = pt.maximum(n_f - 1.0, 0.0) - alts_f
    trans_n = pt.maximum(n_f - 1.0, 0.0)
    
    log_marg_lik = (
        np.log(0.5) 
        + pt.gammaln(alts_f + alpha) 
        + pt.gammaln(stays_f + beta_param) 
        - pt.gammaln(trans_n + alpha + beta_param)
        + pt.gammaln(alpha + beta_param)
        - pt.gammaln(alpha)
        - pt.gammaln(beta_param)
    )
    
    denom = pt.maximum(n_f ** gamma, 1.0)
    return log_marg_lik / denom

def _score_sequence(n, h, alts, w_markov, c_bias, alpha_markov, beta_markov, gamma):
    fair = _scaled_fair_log_prob(n, gamma)
    
    biased = _scaled_betabinomial_log_prob(n, h, c_bias, c_bias, gamma)
    markov = _scaled_markov_betabinomial_log_prob(n, alts, alpha_markov, beta_markov, gamma)
    
    w_bias = 1.0 - w_markov
    
    alternatives = _logsumexp2(
        pt.log(w_bias) + biased,
        pt.log(w_markov) + markov
    )
    
    return fair - alternatives

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    w_markov = pm.Uniform("w_markov", lower=0.01, upper=0.99)
    c_bias = pm.Uniform("c_bias", lower=0.01, upper=50.0)
    alpha_markov = pm.Uniform("alpha_markov", lower=0.01, upper=50.0)
    beta_markov = pm.Uniform("beta_markov", lower=0.01, upper=50.0)
    gamma = pm.Uniform("gamma", lower=0.0, upper=2.0)
    
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)
    
    score_a = _score_sequence(n_a, h_a, alts_a, w_markov, c_bias, alpha_markov, beta_markov, gamma)
    score_b = _score_sequence(n_b, h_b, alts_b, w_markov, c_bias, alpha_markov, beta_markov, gamma)
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
