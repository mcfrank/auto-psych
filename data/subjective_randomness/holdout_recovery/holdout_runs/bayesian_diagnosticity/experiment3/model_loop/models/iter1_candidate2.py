import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALT_SWITCH_PROB = 0.95
STREAK_SWITCH_PROB = 0.15
BIAS_HEAD_PROB = 0.85


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


def _score_sequence(n, h, alts):
    fair = _normalized_iid_log_prob(n, h, 0.5)
    alternating = _normalized_markov_log_prob(n, alts, ALT_SWITCH_PROB)
    streaky = _normalized_markov_log_prob(n, alts, STREAK_SWITCH_PROB)
    bias_h = _normalized_iid_log_prob(n, h, BIAS_HEAD_PROB)
    bias_t = _normalized_iid_log_prob(n, h, 1.0 - BIAS_HEAD_PROB)

    # "Simpler alternative": Instead of a prior-weighted mixture of alternatives,
    # the subject just evaluates the single most likely alternative hypothesis
    # (Generalized Likelihood Ratio Test)
    best_alternative = pt.max(
        pt.stack([alternating, streaky, bias_h, bias_t], axis=0), axis=0
    )

    return fair - best_alternative


with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Score each sequence
    score_a = _score_sequence(n_a, h_a, alts_a)
    score_b = _score_sequence(n_b, h_b, alts_b)

    # Deterministic probability of choosing left
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
