"""PyMC adapter for the (merged) Bayesian-diagnosticity model family.

Randomness as statistical inference (Griffiths & Tenenbaum 2001/2003; Griffiths,
Daniels, Austerweil & Tenenbaum 2018): a sequence looks random when it is better
evidence for a fair coin than for a *regular* generator,

    randomness(x) = log P(x | fair) - log P(x | regular),   P(x | fair) = (1/2)^n.

This single model merges the two former Bayesian seeds. The regular hypothesis is
a mixture (weight ``bias_share``) of:

  * a biased-coin generator (head- or tail-heavy, fixed bias 0.85), capturing
    H/T imbalance; and
  * a motif-complexity process (Griffiths et al. 2018, §6.1) evaluated at the
    canonical parse (n1 = rep_motifs, n2 = alt_motifs):

        log P(x | motif) = (n - n1 - n2)*log δ + (n1 + n2)*log C + (n1 + 2*n2)*log α,
        C = (1-δ)/(2α + 2α²),

    where δ is motif persistence and α penalises motif complexity. This process
    subsumes the old "alternating" and "streaky" Markov alternatives (long runs =
    high persistence, regular alternation = alternation motifs) and carries Falk
    & Konold's DP = n1 + 2*n2 in the α exponent.

The score is not length-normalised; evidence accumulating with length is part of
the account. Free parameters inferred by MCMC: δ, α, bias_share, β, side bias.
"""

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


def _randomness(n, h, rep_motifs, alt_motifs, log_delta, log_c, log_alpha, bias_share):
    log_fair = pt.cast(n, "float64") * np.log(0.5)
    log_motif = _log_motif(n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha)
    log_biased = _log_biased(n, h)
    log_regular = _logsumexp2(
        pt.log(1.0 - bias_share) + log_motif,
        pt.log(bias_share) + log_biased,
    )
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
    bias_share = pm.Uniform("bias_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    log_delta = pt.log(delta)
    log_alpha = pt.log(alpha)
    log_c = pt.log(1.0 - delta) - pt.log(2.0 * alpha + 2.0 * alpha**2)

    score_a = _randomness(
        n_a, h_a, rep_motifs_a, alt_motifs_a, log_delta, log_c, log_alpha, bias_share
    )
    score_b = _randomness(
        n_b, h_b, rep_motifs_b, alt_motifs_b, log_delta, log_c, log_alpha, bias_share
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
