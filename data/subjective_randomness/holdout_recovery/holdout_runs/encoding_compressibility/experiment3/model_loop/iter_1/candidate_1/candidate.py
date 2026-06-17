import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Encoding compressibility model: non-randomness = how much a sequence can be
# compressed relative to a maximally random (max-entropy) sequence.
# Compressibility is measured as KL(observed || uniform) for both the marginal
# distribution and the transition distribution:
#   - zero-order: log(2) - H(p)   where p = proportion of heads
#   - Markov:     log(2) - H(p_alts)  where p_alts = proportion of transitions
# Both are 0 when the sequence looks random (p or p_alts = 0.5) and positive
# when the sequence is structured. This differs from the existing linear models
# by using the entropy function (smooth, symmetric, quadratic near 0.5) rather
# than a linear weight on imbalance or p_alts.

LOG2 = np.log(2.0)

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))

    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Weights are constrained positive: each compressibility signal
    # monotonically increases perceived non-randomness.
    w_markov = pm.HalfNormal("w_markov", sigma=5.0)
    w_zero = pm.HalfNormal("w_zero", sigma=5.0)
    w_run = pm.HalfNormal("w_run", sigma=5.0)
    w_period = pm.HalfNormal("w_period", sigma=5.0)

    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    def kl_from_uniform(p):
        # KL(p || 0.5) in nats = log(2) - H(p) where H is binary entropy.
        # 0 when p = 0.5, positive and increasing as p deviates from 0.5.
        p_c = pt.clip(p, 1e-6, 1.0 - 1e-6)
        h = -(p_c * pt.log(p_c) + (1.0 - p_c) * pt.log(1.0 - p_c))
        return LOG2 - h

    nonrand_a = (
        w_markov * kl_from_uniform(p_alts_a)
        + w_zero * kl_from_uniform(p_a)
        + w_run * max_run_norm_a
        + w_period * periodicity_a
    )
    nonrand_b = (
        w_markov * kl_from_uniform(p_alts_b)
        + w_zero * kl_from_uniform(p_b)
        + w_run * max_run_norm_b
        + w_period * periodicity_b
    )

    # Left chosen when left appears more random (lower non-randomness score).
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(nonrand_b - nonrand_a + side_bias),
    )

    pm.Bernoulli("response", p=p_left, observed=chose_left)
