import numpy as np
import pymc as pm

# Lapse model: on most trials the participant uses structural cues to judge
# randomness; on lapse trials they respond at chance. Attentional lapses are
# common in long binary-choice experiments and can improve predictive fit when
# the deterministic component would otherwise over-commit on hard trials.

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))

    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    nu_val = 4.0
    w_run = pm.StudentT("w_run", nu=nu_val, mu=0.0, sigma=5.0)
    w_period = pm.StudentT("w_period", nu=nu_val, mu=0.0, sigma=5.0)
    w_imbalance = pm.StudentT("w_imbalance", nu=nu_val, mu=0.0, sigma=5.0)

    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Lapse rate: probability the participant responds at random on a given trial.
    # Beta(1, 9) places most prior mass below 0.3, allowing meaningful lapses.
    lapse = pm.Beta("lapse", alpha=1, beta=9)

    score_a = w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a
    score_b = w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b

    p_attend = pm.math.sigmoid((score_a - score_b) + side_bias)

    # Mix attentive response with chance-level responding on lapse trials.
    p_left = pm.Deterministic("p_left", (1.0 - lapse) * p_attend + lapse * 0.5)

    pm.Bernoulli("response", p=p_left, observed=chose_left)
