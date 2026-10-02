"""Personal ideal longest run.

People judge randomness by the longest streak a sequence contains: each person
carries their own ideal length for the longest run of identical flips in a
random sequence (relative to its length), and picks the sequence whose longest
streak is closer to that personal ideal. Ideals differ between people (a
hierarchical, non-centred logit-normal population); nothing else about the
sequence (overall alternation rate, H/T balance) enters the judgement.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments);
# an id at or beyond it fails loudly at indexing.
MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def longest_run_share(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        longest = current = 1
        for x, y in zip(seq, seq[1:]):
            current = current + 1 if x == y else 1
            longest = max(longest, current)
        # 0 = no streak at all (strict alternation), 1 = one single run.
        return (longest - 1) / (len(seq) - 1)

    return {"run_share_a": longest_run_share(sequence_a), "run_share_b": longest_run_share(sequence_b)}


with pm.Model() as model:
    run_share_a = pm.Data("run_share_a", np.zeros(1, dtype="float64"))
    run_share_b = pm.Data("run_share_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population ideal longest-run share (logit scale) and its spread.
    mu_ideal = pm.Normal("mu_ideal", mu=-1.0, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Sensitivity to squared distance from the ideal.
    beta = pm.LogNormal("beta", mu=2.0, sigma=1.0)

    theta = ideal[participant_id]
    score_a = -pt.sqr(run_share_a - theta)
    score_b = -pt.sqr(run_share_b - theta)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
