"""PyMC adapter for the statistical-inference model of subjective randomness.

Griffiths, Daniels, Austerweil & Tenenbaum (2018), "Subjective randomness as
statistical inference" (Cognitive Psychology 103:85-109). A sequence's
randomness is the log-likelihood ratio between a random process (a fair coin)
and a *regular* process:

    randomness(x) = log P(x | random) - log P(x | regular)

with P(x | random) = (1/2)^n for a length-n sequence. The regular process is a
hidden Markov model over motifs (repetition and alternation). For a parse z with
n1 repetition motifs and n2 alternation motifs the paper gives (Section 6.1) the
unnormalized joint

    P(x, z) ∝ δ^(n - n1 - n2) · C^(n1 + n2) · α^(n1 + 2*n2),   C = (1-δ)/(2α+2α²)

where δ is the probability of continuing a motif and α penalizes motif
complexity (longer motifs, which need more states, are exponentially less
likely). Taking P(x | regular) ≈ P(x, z*) at the canonical minimal-description
parse z* (precomputed as the rep_motifs / alt_motifs features), the per-sequence
randomness is closed-form and differentiable in δ and α. Falk & Konold's
Difficulty Predictor, DP = n1 + 2*n2, is the special case carried by the
α exponent. The overall regular-process normalizing constant depends only on
(δ, α), so it cancels in the per-pair comparison below.

Free cognitive parameters inferred by MCMC: δ (motif persistence), α (complexity
penalty), β (choice sensitivity), and a left/right side bias.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _log_regular(n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha):
    """Log P(x | regular) at the canonical parse (Griffiths et al. 2018, §6.1)."""
    n_f = pt.cast(n, "float64")
    n1 = pt.cast(rep_motifs, "float64")
    n2 = pt.cast(alt_motifs, "float64")
    stays = n_f - n1 - n2  # within-motif continuations; >= 0 by construction
    return stays * log_delta + (n1 + n2) * log_c + (n1 + 2.0 * n2) * log_alpha


def _randomness(n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha):
    log_random = pt.cast(n, "float64") * np.log(0.5)
    return log_random - _log_regular(
        n, rep_motifs, alt_motifs, log_delta, log_c, log_alpha
    )


with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    delta = pm.Uniform("delta", lower=0.01, upper=0.99)
    alpha = pm.Uniform("alpha", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    log_delta = pt.log(delta)
    log_alpha = pt.log(alpha)
    log_c = pt.log(1.0 - delta) - pt.log(2.0 * alpha + 2.0 * alpha**2)

    randomness_a = _randomness(
        n_a, rep_motifs_a, alt_motifs_a, log_delta, log_c, log_alpha
    )
    randomness_b = _randomness(
        n_b, rep_motifs_b, alt_motifs_b, log_delta, log_c, log_alpha
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (randomness_a - randomness_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
