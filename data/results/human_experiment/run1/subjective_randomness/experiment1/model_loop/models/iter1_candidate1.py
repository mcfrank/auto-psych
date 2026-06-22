"""
People judge the randomness of a sequence by comparing it to a mental prototype that embodies the expected natural variability of a stochastic process, actively distrusting sequences that appear artificially "too perfect." Rather than expecting an exactly balanced sequence, they recognize that flawless balance is statistically rare and thus expect a moderate, typical degree of outcome imbalance alongside a specific alternation rate. Sequences are perceived as more random when their degree of imbalance and alternation rate have a smaller squared deviation from these subjective, non-zero ideal expectations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Features from responses
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))

    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free parameters for the subjective prototype
    ideal_imbalance = pm.Beta("ideal_imbalance", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)

    # Weights for the deviations (also serve as inverse temperature)
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)

    # Calculate squared deviations from the subjective ideals
    dev_a = w_imb * pt.square(imbalance_a - ideal_imbalance) + w_alt * pt.square(
        p_alts_a - ideal_alt
    )
    dev_b = w_imb * pt.square(imbalance_b - ideal_imbalance) + w_alt * pt.square(
        p_alts_b - ideal_alt
    )

    # Lower deviation means it is closer to the prototype (more random)
    p_left_raw = pm.math.sigmoid(dev_b - dev_a)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
