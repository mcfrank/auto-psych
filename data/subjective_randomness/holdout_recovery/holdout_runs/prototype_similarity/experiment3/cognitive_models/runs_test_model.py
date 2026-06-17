"""Runs-test model: randomness judged via the Wald-Wolfowitz z-score.

People implicitly test whether a sequence's run count R is consistent with a
fair-coin process by computing a standardized deviate:
    z = (R - E[R]) / sqrt(Var[R])
where E[R] = 2ht/n + 1 and Var[R] = 2ht(2ht - n) / (n²(n-1)) follow from
combinatorics (Wald & Wolfowitz 1940). Unlike the statistical-expectation model
(inner_loop_model) which normalizes by n, this model normalizes by the theoretical
standard deviation of R — so the same raw deviation carries more weight for
balanced sequences than for imbalanced ones. Asymmetric streak_k penalizes
below-expected runs (streakiness) more than above-expected (over-alternation),
consistent with gambler's-fallacy sensitivity.
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

    streak_k = pm.HalfNormal("streak_k", sigma=2.0)
    imbalance_weight = pm.Uniform("imbalance_weight", lower=0.0, upper=1.0)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    def score(n, h, alts, imbalance):
        n_f = pt.cast(n, "float64")
        h_f = pt.cast(h, "float64")
        t_f = n_f - h_f
        alts_f = pt.cast(alts, "float64")

        R = alts_f + 1.0
        E_R = 2.0 * h_f * t_f / (n_f + 1e-8) + 1.0
        var_R = (2.0 * h_f * t_f * (2.0 * h_f * t_f - n_f)) / (
            n_f * n_f * pt.maximum(n_f - 1.0, 1e-8) + 1e-8
        )
        std_R = pt.sqrt(pt.maximum(var_R, 1e-8))
        z = (R - E_R) / std_R

        z_penalty = streak_k * pt.maximum(-z, 0.0) + pt.maximum(z, 0.0)
        return -(imbalance_weight * imbalance + (1.0 - imbalance_weight) * z_penalty)

    score_a = score(n_a, h_a, alts_a, imbalance_a)
    score_b = score(n_b, h_b, alts_b, imbalance_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
