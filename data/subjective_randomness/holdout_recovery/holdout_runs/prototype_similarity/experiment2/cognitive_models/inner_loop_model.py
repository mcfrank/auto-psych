"""Statistical-expectation model: randomness judged relative to what a fair coin predicts.

All existing models compare sequences to a fixed learned prototype (theta_alt).
This model uses a different cognitive mechanism: people implicitly compare the
observed number of alternations to the expected number from a fair coin given
the sequence's own length and balance. A sequence feels non-random when its
alternation count deviates from this per-sequence expectation.

From the classical runs test, for a sequence with h heads and t = n - h tails
drawn independently at p = 0.5, the expected number of alternations is
    E[alts] = 2 * h * t / n.
A deviation above expectation flags over-alternation (gambler's-fallacy
sensitivity), and a deficit flags streakiness — each penalised asymmetrically
via streak_k.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Asymmetric sensitivity: streaky sequences (below expected alternations)
    # penalised more than over-alternating ones (gambler's fallacy direction).
    # streak_k = 1 gives a symmetric penalty.
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)

    # How much weight balance gets relative to alternation deviation.
    imbalance_weight = pm.Uniform("imbalance_weight", lower=0.0, upper=1.0)

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    def score(n, h, alts, imbalance):
        n_f = pt.cast(n, "float64")
        h_f = pt.cast(h, "float64")
        t_f = n_f - h_f
        alts_f = pt.cast(alts, "float64")

        # Expected alternations for a fair-coin sequence with this balance.
        expected_alts = 2.0 * h_f * t_f / (n_f + 1e-8)

        # Signed deviation; positive = over-alternating, negative = streaky.
        deviation = alts_f - expected_alts

        # Asymmetric penalty, normalised by length so sequences of different
        # lengths are comparable.
        alt_penalty = (
            streak_k * pt.maximum(-deviation, 0.0)
            + pt.maximum(deviation, 0.0)
        ) / (n_f + 1e-8)

        return -(imbalance_weight * imbalance + (1.0 - imbalance_weight) * alt_penalty)

    score_a = score(n_a, h_a, alts_a, imbalance_a)
    score_b = score(n_b, h_b, alts_b, imbalance_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
