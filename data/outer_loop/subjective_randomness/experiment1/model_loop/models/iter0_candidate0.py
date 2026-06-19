"""
People judge which sequence looks more random by computing Bayesian
diagnosticity against a single implicit alternative: a repetitive (streaky)
generator that rarely alternates. The sequence whose log-likelihood ratio —
fair coin versus repetitive generator — is higher is chosen as more random.
The repetitive generator's alternation probability is a free parameter
inferred from behavior, expected to sit well below 0.5.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    # Free parameter: alternation probability of the implicit repetitive generator.
    # Prior centered at logit(-1.5) ≈ 0.18, favouring a streaky generator.
    logit_p_rep = pm.Normal("logit_p_rep", mu=-1.5, sigma=1.0)
    p_rep = pm.Deterministic("p_rep", pm.math.sigmoid(logit_p_rep))
    p_safe = pt.clip(p_rep, 1e-6, 1 - 1e-6)

    # Sensitivity: scales how strongly the LLR difference drives the choice.
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Log-likelihood ratio: fair coin vs. repetitive generator (transition model).
    # LLR = (n-1)*log(0.5) - alts*log(p_rep) - (n-1-alts)*log(1-p_rep)
    log05 = np.log(0.5)
    log_lr_a = (
        (n_a - 1) * log05
        - alts_a * pt.log(p_safe)
        - (n_a - 1 - alts_a) * pt.log(1 - p_safe)
    )
    log_lr_b = (
        (n_b - 1) * log05
        - alts_b * pt.log(p_safe)
        - (n_b - 1 - alts_b) * pt.log(1 - p_safe)
    )

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (log_lr_a - log_lr_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
