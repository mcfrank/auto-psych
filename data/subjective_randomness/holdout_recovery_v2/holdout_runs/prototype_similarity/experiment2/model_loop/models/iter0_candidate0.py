"""
People judge sequences by how diagnostic they are of a fair-coin process against three
salient non-random alternatives — alternating, biased, and streaky. The alternating
generator has a canonical switch probability of 0.95, but the streaky generator's
characteristic persistence (its switch probability) is learned from the data rather
than fixed at a canonical value.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


ALT_SWITCH_PROB = 0.95
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
        np.log(0.5) + alts_f * pt.log(switch_prob) + stays_f * pt.log(1.0 - switch_prob)
    ) / denom


def _score_sequence(n, h, alts, streak_switch_prob, alt_prior, bias_share):
    fair = _normalized_iid_log_prob(n, h, 0.5)
    alternating = _normalized_markov_log_prob(n, alts, ALT_SWITCH_PROB)
    streaky = _normalized_markov_log_prob(n, alts, streak_switch_prob)
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
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Learned switch probability for the "streaky" prototype generator.
    # Beta(2, 10) is weakly informative: prior mean ~0.17, near the canonical 0.15.
    streak_switch_prob = pm.Beta("streak_switch_prob", alpha=2.0, beta=10.0)
    alt_prior = pm.Uniform("alt_prior", lower=0.01, upper=0.98)
    bias_share = pm.Uniform("bias_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)

    score_a = _score_sequence(n_a, h_a, alts_a, streak_switch_prob, alt_prior, bias_share)
    score_b = _score_sequence(n_b, h_b, alts_b, streak_switch_prob, alt_prior, bias_share)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b)),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
