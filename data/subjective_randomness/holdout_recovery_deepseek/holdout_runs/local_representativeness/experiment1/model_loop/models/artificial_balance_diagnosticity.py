"""People judge the randomness of a sequence by computing its Bayesian diagnosticity—the log-likelihood ratio of a fair coin versus a subjective 'regular' generative process. We refine this model by adding an 'artificial balance' component to the regular hypothesis, representing a generator that intentionally targets an exact equal count of heads and tails. This explains why people strongly penalize perfectly symmetric sequences, recognizing them as suspiciously contrived rather than naturally random."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

BIAS_HEAD_PROB = 0.85

def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))

def _log_motif(n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha):
    n_f = pt.cast(n, "float64")
    n1 = pt.cast(rep_motifs, "float64")
    n2 = pt.cast(alt_motifs, "float64")
    stays = n_f - n1 - n2  # within-motif continuations; >= 0 by construction
    return stays * log_delta + (n1 + n2) * log_c + (n1 + 2.0 * n2) * log_alpha

def _log_biased(n, h):
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    tails_f = n_f - h_f
    head_heavy = h_f * np.log(BIAS_HEAD_PROB) + tails_f * np.log(1.0 - BIAS_HEAD_PROB)
    tail_heavy = h_f * np.log(1.0 - BIAS_HEAD_PROB) + tails_f * np.log(BIAS_HEAD_PROB)
    return _logsumexp2(np.log(0.5) + head_heavy, np.log(0.5) + tail_heavy)

def _log_balanced(n, h, gamma):
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    
    # log P(h | balanced) = -gamma * |h - n/2| - log_Z
    max_len = 50
    k = pt.arange(max_len + 1, dtype="float64")
    k_exp = pt.expand_dims(k, 1)
    n_f_exp = pt.expand_dims(n_f, 0)
    
    mask = pt.cast(k_exp <= n_f_exp, "float64")
    unnormalized = pt.exp(-gamma * pt.abs(k_exp - n_f_exp / 2.0)) * mask
    Z = pt.sum(unnormalized, axis=0)
    
    log_p_h = -gamma * pt.abs(h_f - n_f / 2.0) - pt.log(Z)
    
    # log (1 / (n choose h))
    log_choose = pt.gammaln(n_f + 1.0) - pt.gammaln(h_f + 1.0) - pt.gammaln(n_f - h_f + 1.0)
    
    return log_p_h - log_choose

def _randomness(n, h, rep_motifs, alt_motifs, log_delta, log_c, log_alpha, weights, gamma):
    log_fair = pt.cast(n, "float64") * np.log(0.5)
    
    log_m = _log_motif(n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha)
    log_bi = _log_biased(n, h)
    log_bal = _log_balanced(n, h, gamma)
    
    log_w_m = pt.log(weights[0])
    log_w_bi = pt.log(weights[1])
    log_w_bal = pt.log(weights[2])
    
    components = pt.stack([
        log_w_m + log_m,
        log_w_bi + log_bi,
        log_w_bal + log_bal
    ], axis=0)
    
    log_regular = pt.logsumexp(components, axis=0)
    return log_fair - log_regular

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    delta = pm.Uniform("delta", lower=0.01, upper=0.99)
    alpha = pm.Uniform("alpha", lower=0.01, upper=0.99)
    weights = pm.Dirichlet("weights", a=np.ones(3))
    gamma = pm.HalfNormal("gamma", sigma=5.0)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    log_delta = pt.log(delta)
    log_alpha = pt.log(alpha)
    log_c = pt.log(1.0 - delta) - pt.log(2.0 * alpha + 2.0 * alpha**2)

    score_a = _randomness(
        n_a, h_a, rep_motifs_a, alt_motifs_a, log_delta, log_c, log_alpha, weights, gamma
    )
    score_b = _randomness(
        n_b, h_b, rep_motifs_b, alt_motifs_b, log_delta, log_c, log_alpha, weights, gamma
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
