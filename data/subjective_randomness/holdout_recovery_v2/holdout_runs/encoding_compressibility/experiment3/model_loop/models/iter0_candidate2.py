"""
People judge a sequence as more random when its head proportion is closer to 0.5,
because a balanced sequence looks most consistent with a fair coin. On some fraction
of trials they lapse and respond randomly. The core mechanism is pure head-proportion
balance, extended by a single lapse-rate parameter for inattentive trials.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — proportion of heads in each sequence.
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Sensitivity to head-balance difference.
    beta = pm.HalfNormal("beta", sigma=5.0)
    # Lapse rate: probability of responding randomly on any given trial.
    lapse = pm.Beta("lapse", alpha=1, beta=10)

    # Balance score: negative distance from 0.5; higher = more balanced = more random.
    balance_a = -pt.abs(p_a - 0.5)
    balance_b = -pt.abs(p_b - 0.5)

    p_core = pm.math.sigmoid(beta * (balance_a - balance_b))
    p_left = pm.Deterministic("p_left", (1.0 - lapse) * p_core + lapse * 0.5)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
