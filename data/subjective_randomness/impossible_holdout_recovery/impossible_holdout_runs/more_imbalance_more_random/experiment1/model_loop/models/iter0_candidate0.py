"""
People judge sequence randomness based on the Bayesian diagnosticity of a fair coin compared to a single salient alternative: a "streaky" Markov generator that has an innate tendency to repeat previous outcomes. Sequences that are more likely under the fair coin than the streaky alternative are perceived as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Feature columns for sequence A
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    # Feature columns for sequence B
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Subjective probability of alternation for the "streaky" alternative.
    # We use a Beta prior that naturally favors lower values (streaks).
    p_alt_streaky = pm.Beta("p_alt_streaky", alpha=2.0, beta=5.0)
    
    # Sensitivity parameter
    tau = pm.HalfNormal("tau", sigma=2.0)
    
    # Clip for numerical safety in logs
    p_alt_s_safe = pt.clip(p_alt_streaky, 1e-6, 1.0 - 1e-6)
    
    # Log-likelihood under a fair coin.
    # The first flip is 0.5, and each of the (n-1) transitions is 0.5.
    ll_fair_a = n_a * pt.log(0.5)
    ll_fair_b = n_b * pt.log(0.5)
    
    # Log-likelihood under the streaky alternative.
    # First flip is 0.5. There are alts transitions of p_alt_s_safe,
    # and (n - 1 - alts) transitions of (1 - p_alt_s_safe).
    ll_streaky_a = pt.log(0.5) + alts_a * pt.log(p_alt_s_safe) + (n_a - 1 - alts_a) * pt.log(1.0 - p_alt_s_safe)
    ll_streaky_b = pt.log(0.5) + alts_b * pt.log(p_alt_s_safe) + (n_b - 1 - alts_b) * pt.log(1.0 - p_alt_s_safe)
    
    # Diagnosticity (log Bayes factor) for fair over streaky
    # Positive values mean fair is more likely.
    diag_a = ll_fair_a - ll_streaky_a
    diag_b = ll_fair_b - ll_streaky_b
    
    # Choice is driven by the difference in diagnosticity.
    # If diag_a > diag_b, sequence A is more random-looking, p_left should be > 0.5.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (diag_a - diag_b)))
    
    # Ensure numerical safety
    p_left_safe = pt.clip(p_left, 1e-6, 1.0 - 1e-6)
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
