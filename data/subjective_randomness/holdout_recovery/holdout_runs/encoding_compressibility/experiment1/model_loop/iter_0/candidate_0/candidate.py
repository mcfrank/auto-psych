import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _logsumexp2(a, b):
    m = pt.maximum(a, b)
    return m + pt.log(pt.exp(a - m) + pt.exp(b - m))


def _normalized_iid_log_prob(n, h, p_heads):
    n_f = pt.cast(n, "float64")
    h_f = pt.cast(h, "float64")
    tails_f = n_f - h_f
    denom = pt.maximum(n_f, 1.0)
    return (h_f * pt.log(p_heads) + tails_f * pt.log(1.0 - p_heads)) / denom


def _normalized_markov_log_prob(n, alts, switch_prob):
    n_f = pt.cast(n, "float64")
    alts_f = pt.cast(alts, "float64")
    stays_f = pt.maximum(n_f - 1.0, 0.0) - alts_f
    denom = pt.maximum(n_f, 1.0)
    return (
        np.log(0.5) + alts_f * pt.log(switch_prob) + stays_f * pt.log(1.0 - switch_prob)
    ) / denom


def _score_sequence(
    n, h, alts, alt_prior, bias_share, alt_switch_prob, streak_switch_prob, bias_head_prob
):
    fair = _normalized_iid_log_prob(n, h, 0.5)
    alternating = _normalized_markov_log_prob(n, alts, alt_switch_prob)
    streaky = _normalized_markov_log_prob(n, alts, streak_switch_prob)
    bias_h = _normalized_iid_log_prob(n, h, bias_head_prob)
    bias_t = _normalized_iid_log_prob(n, h, 1.0 - bias_head_prob)
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

    alt_prior = pm.Uniform("alt_prior", lower=0.01, upper=0.98)
    bias_share = pm.Uniform("bias_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)
    
    # Refinement: Infer subjective probabilities for the alternative hypotheses
    # rather than using hardcoded values.
    alt_switch_prob = pm.Uniform("alt_switch_prob", lower=0.5, upper=0.99)
    streak_switch_prob = pm.Uniform("streak_switch_prob", lower=0.01, upper=0.5)
    bias_head_prob = pm.Uniform("bias_head_prob", lower=0.5, upper=0.99)

    score_a = _score_sequence(
        n_a, h_a, alts_a, alt_prior, bias_share, alt_switch_prob, streak_switch_prob, bias_head_prob
    )
    score_b = _score_sequence(
        n_b, h_b, alts_b, alt_prior, bias_share, alt_switch_prob, streak_switch_prob, bias_head_prob
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
