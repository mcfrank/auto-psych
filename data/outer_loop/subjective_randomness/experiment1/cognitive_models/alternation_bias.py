"""Observers judge randomness by alternation proportion — the fraction of adjacent
positions that differ (H→T or T→H transitions). People expect fair coins to
alternate more than they actually do, so they prefer the more alternating
sequence as more random. Softmax decision rule with a single temperature parameter.

Captures the classic gambler's-fallacy / negative recency bias documented by
Kahneman & Tversky (1972) and operationalised in the similarity model of
Griffiths & Tenenbaum (2001)."""
import numpy as np
import pymc as pm

with pm.Model() as model:
    # Alternation proportions: alts / (n - 1), range [0, 1].
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=2.0)  # softmax temperature

    # Prefer whichever sequence has the higher alternation proportion.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_alts_a - p_alts_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
