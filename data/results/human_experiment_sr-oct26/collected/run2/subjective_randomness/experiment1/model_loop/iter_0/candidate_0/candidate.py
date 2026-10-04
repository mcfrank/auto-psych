"""Personal Markov coin belief.

Each person carries their own subjective model of what a fair coin does: a
belief about how often a random coin switches between H and T from one flip to
the next. They judge which sequence is more random by how probable each
sequence would be under their own believed coin, choosing between the two in
proportion to those probabilities (Luce choice), so people differ in how
strongly, and in which direction, alternation makes a sequence look random.

Under a believed switch probability q, log P(seq) = log 0.5
+ switches*log q + repeats*log(1-q). For equal-length sequences the Luce
choice probability is sigmoid((switches_a - switches_b) * logit(q_i)).
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switches(seq):
        seq = seq.strip().upper()
        return float(sum(1 for x, y in zip(seq, seq[1:]) if x != y))

    return {"switch_diff": switches(sequence_a) - switches(sequence_b)}


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of believed switch log-odds, logit(q_i).
    mu = pm.Normal("mu", mu=0.0, sigma=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=400)
    belief_logit = pm.Deterministic("belief_logit", mu + sigma * z)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(switch_diff * belief_logit[participant_id])
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
