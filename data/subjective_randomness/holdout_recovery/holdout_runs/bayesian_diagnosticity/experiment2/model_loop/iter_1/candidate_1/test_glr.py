import numpy as np
import pytensor.tensor as pt

def glr_bias(n, h):
    p_hat = h / n
    # avoid log(0)
    p_hat = pt.clip(p_hat, 1e-5, 1.0 - 1e-5)
    log_fair = n * np.log(0.5)
    log_biased = h * pt.log(p_hat) + (n - h) * pt.log(1.0 - p_hat)
    return log_fair - log_biased

def glr_alt(n, alts):
    n_trans = pt.maximum(n - 1, 1e-5)
    p_alt_hat = alts / n_trans
    p_alt_hat = pt.clip(p_alt_hat, 1e-5, 1.0 - 1e-5)
    log_fair = n_trans * np.log(0.5)
    log_alt = alts * pt.log(p_alt_hat) + (n_trans - alts) * pt.log(1.0 - p_alt_hat)
    return log_fair - log_alt
