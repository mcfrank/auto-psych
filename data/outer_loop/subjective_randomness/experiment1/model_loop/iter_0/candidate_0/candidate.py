"""Combines two cues for perceived randomness: alternation proportion
(negative recency bias, Kahneman & Tversky 1972) and max run length
(long streaks feel non-random). Each cue has its own weight so the
model can estimate their relative importance.

Extends alternation_bias by adding the max-run cue with an independent
weight, allowing partial credit for both signals."""

import numpy as np
import pymc as pm

with pm.Model() as model:
    # Cue 1: alternation proportions — higher means more alternating, more random-looking.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Cue 2: max run lengths — shorter max run means more random-looking.
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Independent weights for each cue.
    w_alts = pm.HalfNormal("w_alts", sigma=2.0)
    w_runs = pm.HalfNormal("w_runs", sigma=2.0)

    # Utility difference for A over B: positive → prefer A (chose_left).
    # Alternation: prefer A when p_alts_a > p_alts_b.
    # Runs: prefer A when max_run_a < max_run_b (reversed sign).
    util = w_alts * (p_alts_a - p_alts_b) + w_runs * (max_run_b - max_run_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(util))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
