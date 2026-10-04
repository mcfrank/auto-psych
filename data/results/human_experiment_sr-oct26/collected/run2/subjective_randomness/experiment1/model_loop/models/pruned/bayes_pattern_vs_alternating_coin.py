"""Bayesian pattern detection with a distorted (over-alternating) coin belief.

People judge randomness by Bayesian model comparison: a sequence looks random
to the extent it is better explained by a random coin than by a noisy
repeating pattern (a template of period 1-4, e.g. HHHH, HTHT, HHTT, copied
with occasional errors, the error rate integrated out under a uniform prior).
The single distortion is in the random-coin hypothesis: each person believes a
fair coin switches sides at their own rate (often above one half), so
alternation is evidence for randomness until the sequence becomes regular
enough to be better explained by a repeating template.
"""
import itertools
import math

import numpy as np
import pymc as pm

MAX_PERIOD = 4


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _pattern_log_ml(seq):
    """Log marginal likelihood under the noisy-template generator."""
    n = len(seq)
    terms = []
    for k in range(1, MAX_PERIOD + 1):
        templates = list(itertools.product("HT", repeat=k))
        for t in templates:
            m = sum(1 for i, c in enumerate(seq) if c != t[i % k])
            # uniform prior over period, then over templates of that period
            terms.append(-math.log(MAX_PERIOD) - math.log(len(templates)) + _log_beta(m + 1, n - m + 1))
    mx = max(terms)
    return mx + math.log(sum(math.exp(x - mx) for x in terms))


def compute_features(sequence_a, sequence_b):
    def feats(seq):
        seq = seq.strip().upper()
        s = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return float(s), float(len(seq) - 1 - s), _pattern_log_ml(seq)

    sa, ra, pa = feats(sequence_a)
    sb, rb, pb = feats(sequence_b)
    return {
        "switches_a": sa, "repeats_a": ra, "pattern_lml_a": pa,
        "switches_b": sb, "repeats_b": rb, "pattern_lml_b": pb,
    }


N_SLOTS = 400

with pm.Model() as model:
    switches_a = pm.Data("switches_a", np.zeros(1, dtype="float64"))
    repeats_a = pm.Data("repeats_a", np.zeros(1, dtype="float64"))
    pattern_a = pm.Data("pattern_lml_a", np.zeros(1, dtype="float64"))
    switches_b = pm.Data("switches_b", np.zeros(1, dtype="float64"))
    repeats_b = pm.Data("repeats_b", np.zeros(1, dtype="float64"))
    pattern_b = pm.Data("pattern_lml_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Each person's believed switch probability of a fair coin (logit scale).
    mu_switch = pm.Normal("mu_switch", mu=0.0, sigma=1.0)
    sigma_switch = pm.HalfNormal("sigma_switch", sigma=1.0)
    z_switch = pm.Normal("z_switch", mu=0.0, sigma=1.0, shape=N_SLOTS)
    logit_phi = (mu_switch + sigma_switch * z_switch)[participant_id]

    # Decision sensitivity to the log posterior odds (1 = exact Bayesian).
    beta = pm.HalfNormal("beta", sigma=2.0)

    log_phi = -pm.math.log1pexp(-logit_phi)
    log_1m_phi = -pm.math.log1pexp(logit_phi)
    coin_a = switches_a * log_phi + repeats_a * log_1m_phi
    coin_b = switches_b * log_phi + repeats_b * log_1m_phi
    odds_a = coin_a - pattern_a
    odds_b = coin_b - pattern_b

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (odds_a - odds_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
