"""
People judge a sequence as more random when its Shannon entropy is higher — when
its head proportion is closest to 0.5 in information-theoretic terms. Entropy
penalizes extreme imbalance disproportionately more than mild imbalance, making
this a higher-variance single-cue model that refines linear head balance.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Head proportions for each sequence — names match responses CSV columns.
    p_a = pm.Data("p_a", np.zeros(1))
    p_b = pm.Data("p_b", np.zeros(1))

    # Clamp to avoid log(0) at boundary values.
    p_a_safe = pt.clip(p_a, 1e-6, 1 - 1e-6)
    p_b_safe = pt.clip(p_b, 1e-6, 1 - 1e-6)

    # Shannon entropy for each sequence: H(p) = -p*log(p) - (1-p)*log(1-p).
    H_a = -p_a_safe * pt.log(p_a_safe) - (1 - p_a_safe) * pt.log(1 - p_a_safe)
    H_b = -p_b_safe * pt.log(p_b_safe) - (1 - p_b_safe) * pt.log(1 - p_b_safe)

    # Inverse temperature: how sharply participants discriminate on entropy.
    beta = pm.HalfNormal("beta", sigma=5.0)

    # Person prefers sequence A (left) when H_a > H_b.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (H_a - H_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
