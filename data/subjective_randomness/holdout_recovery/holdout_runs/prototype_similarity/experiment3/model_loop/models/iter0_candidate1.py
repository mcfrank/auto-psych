"""Max-violation model: randomness judged by the single worst deviation.

Additive models assume people continuously integrate evidence across feature
dimensions (weighted-sum strategy). This model tests the opposite: people
identify the most salient deviation from randomness and base their judgment
on that lone signal -- a "smoking gun" detection strategy.

The non-randomness penalty for a sequence equals MAX(alt_deviation,
imbalance) rather than a weighted sum. One glaring violation is sufficient to
deem a sequence non-random; a second violation adds nothing. This is
qualitatively different from all additive models in the current landscape.

The alternation deviation uses the same asymmetric streak penalty as the
top-ranked asymmetric_alternation_prototype, so differences in ELPD isolate
the max-vs-sum aggregation question.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate that feels most random.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Asymmetric streak sensitivity (>1: streakiness penalised more).
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)

    # Rescale the alternation deviation to be comparable with imbalance [0,1].
    w_alt = pm.HalfNormal("w_alt", sigma=2.0)

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    def asymmetric_dev(p_alts):
        return (
            streak_k * pt.maximum(theta_alt - p_alts, 0.0)
            + pt.maximum(p_alts - theta_alt, 0.0)
        )

    # Non-randomness = worst single violation, not sum of violations.
    penalty_a = pt.maximum(w_alt * asymmetric_dev(p_alts_a), imbalance_a)
    penalty_b = pt.maximum(w_alt * asymmetric_dev(p_alts_b), imbalance_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (-penalty_a - -penalty_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
