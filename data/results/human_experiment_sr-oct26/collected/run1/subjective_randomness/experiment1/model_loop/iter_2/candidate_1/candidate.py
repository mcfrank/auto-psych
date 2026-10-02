"""Personal ideal for the most lopsided stretch.

People judge a coin-flip sequence by its single most lopsided stretch: the one
contiguous run of flips in which heads most outnumber tails (or vice versa).
The size of that worst local excess (the range of the running heads-minus-tails
count, divided by the sequence length) is compared with each person's own ideal
for it; the sequence closer to the personal ideal looks more random. Nothing
else about the sequence matters beyond that one maximum over sub-sequences.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments);
# an id at or beyond it fails loudly at indexing.
MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def max_lopsided(seq):
        seq = seq.strip().upper()
        if len(seq) < 2 or set(seq) - {"H", "T"}:
            raise ValueError(f"bad sequence: {seq!r}")
        # Max over contiguous stretches of |#H - #T| = range of the running walk.
        walk = [0]
        for c in seq:
            walk.append(walk[-1] + (1 if c == "H" else -1))
        return (max(walk) - min(walk)) / len(seq)

    return {"lopsided_a": max_lopsided(sequence_a), "lopsided_b": max_lopsided(sequence_b)}


with pm.Model() as model:
    lopsided_a = pm.Data("lopsided_a", np.zeros(1, dtype="float64"))
    lopsided_b = pm.Data("lopsided_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population ideal worst-stretch size (logit scale) and its spread.
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Sensitivity to squared distance from the ideal.
    beta = pm.LogNormal("beta", mu=2.0, sigma=1.0)

    theta = ideal[participant_id]
    score_a = -pt.sqr(lopsided_a - theta)
    score_b = -pt.sqr(lopsided_b - theta)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
