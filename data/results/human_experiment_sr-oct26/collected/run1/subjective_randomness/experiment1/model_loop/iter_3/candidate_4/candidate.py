"""Personal lapse, personal ideal alternation, plus longest-streak aversion.

Refinement of `personal_lapse_ideal_alternation`: people judge a sequence as
more random the closer its alternation rate lies to their own ideal switching
rate, guess on a person-specific share of trials, and, as the single change,
when engaged also penalise the sequence's longest run of identical flips
(as a fraction of its length), beyond what the alternation rate shows.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def longest_run_frac(seq):
        seq = seq.strip().upper()
        best = run = 1
        for x, y in zip(seq, seq[1:]):
            run = run + 1 if x == y else 1
            best = max(best, run)
        return best / len(seq)

    return {
        "alt_rate_a": alternation_rate(sequence_a),
        "alt_rate_b": alternation_rate(sequence_b),
        "run_frac_a": longest_run_frac(sequence_a),
        "run_frac_b": longest_run_frac(sequence_b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    run_frac_a = pm.Data("run_frac_a", np.zeros(1, dtype="float64"))
    run_frac_b = pm.Data("run_frac_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.0, sigma=0.5)
    # Shared penalty on the longest streak's share of the sequence.
    gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

    # Personal lapse (guessing) rate (non-centred logit-normal population).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - gamma * run_frac_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - gamma * run_frac_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
