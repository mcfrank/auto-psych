"""Cohesion-referenced exemplar similarity.

People judge a target by its similarity to the remembered examples (on
magnitude, final digit and parity), but compare that similarity not to a fixed
standard but to how similar the examples are to one another ("does this number
fit in with the examples as well as they fit with each other?"). Coherent sets
set a high bar (narrow generalization, low endorsement); scattered sets set a
low bar, so overall willingness to say yes varies strongly between sets.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
PAIRS = list(itertools.combinations(range(MAXK), 2))


def _diffs(a, b):
    return abs(a - b) / 10.0, float(a % 10 != b % 10), float(a % 2 != b % 2)


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    n = len(ex)
    out = {}
    for k in range(MAXK):
        d, u, p = _diffs(t, ex[k]) if k < n else (0.0, 0.0, 0.0)
        out[f"t_dist_{k}"], out[f"t_unit_{k}"], out[f"t_par_{k}"] = d, u, p
        out[f"t_present_{k}"] = 1.0 if k < n else 0.0
    for j, (a, b) in enumerate(PAIRS):
        ok = b < n
        d, u, p = _diffs(ex[a], ex[b]) if ok else (0.0, 0.0, 0.0)
        out[f"p_dist_{j}"], out[f"p_unit_{j}"], out[f"p_par_{j}"] = d, u, p
        out[f"p_present_{j}"] = 1.0 if ok else 0.0
    out["singleton"] = 1.0 if n == 1 else 0.0
    return out


def _stack(prefix, m):
    return pt.stack([pm.Data(f"{prefix}_{k}", np.zeros(1, dtype="float64")) for k in range(m)], axis=1)


with pm.Model() as model:
    td, tu, tp, tpres = (_stack(f"t_{s}", MAXK) for s in ("dist", "unit", "par", "present"))
    pd_, pu, pp, ppres = (_stack(f"p_{s}", len(PAIRS)) for s in ("dist", "unit", "par", "present"))
    singleton = pm.Data("singleton", np.zeros(1, dtype="float64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    log_c_parity = pm.Normal("log_c_parity", mu=0.0, sigma=1.5)
    c_mag, c_unit, c_par = pt.exp(log_c_mag), pt.exp(log_c_unit), pt.exp(log_c_parity)
    # How strongly the bar rises with the examples' mutual similarity
    # (on the logit scale directly, so it does not trade off with beta).
    cohesion_weight = pm.HalfNormal("cohesion_weight", sigma=1.0)
    # Logit offset for single-example sets (no mutual similarity to read off).
    singleton_offset = pm.Normal("singleton_offset", mu=0.0, sigma=1.5)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    # Target: mean similarity to the examples (log scale).
    log_sim_t = -c_mag * td - c_unit * tu - c_par * tp
    log_sim_t = pt.switch(tpres > 0.5, log_sim_t, -1e3)
    n_ex = pt.sum(tpres, axis=1)
    log_fit_t = pm.math.logsumexp(log_sim_t, axis=1).flatten() - pt.log(n_ex)

    # Set cohesion: mean pairwise similarity among the examples (log scale).
    log_sim_p = -c_mag * pd_ - c_unit * pu - c_par * pp
    log_sim_p = pt.switch(ppres > 0.5, log_sim_p, -1e3)
    n_pairs = pt.maximum(pt.sum(ppres, axis=1), 1.0)
    log_coh = pm.math.logsumexp(log_sim_p, axis=1).flatten() - pt.log(n_pairs)

    # The bar: higher cohesion raises the criterion the target must clear.
    bar = pt.switch(singleton > 0.5, -singleton_offset, cohesion_weight * log_coh)
    logit = alpha + beta * log_fit_t - bar
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
