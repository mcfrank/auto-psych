import numpy as np
import pytensor.tensor as pt

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

def _normalized_betabinomial_log_prob(n, k, alpha, beta_param):
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
    denom = pt.maximum(n_f, 1.0)
    return log_marg_lik / denom

def _normalized_markov_betabinomial_log_prob(n, alts, alpha, beta_param):
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
    denom = pt.maximum(n_f, 1.0)
    return log_marg_lik / denom

# test values
n = pt.as_tensor_variable(np.array([10]))
h = pt.as_tensor_variable(np.array([5]))
alts = pt.as_tensor_variable(np.array([5]))

b = _normalized_betabinomial_log_prob(n, h, 1.0, 1.0)
m = _normalized_markov_betabinomial_log_prob(n, alts, 1.0, 1.0)

print("success!")
