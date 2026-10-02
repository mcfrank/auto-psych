"""Personal switch-belief model.

Each person holds their own subjective model of a fair coin as a process that
switches between heads and tails with a personal probability, and judges as
more random whichever sequence is more probable under that belief. Believed
switch rates differ between people, so the same pair can be judged in opposite
directions by different participants.

Under a switch process with probability q, a length-n sequence with s switches
has log-probability s*log(q) + (n-1-s)*log(1-q) (+ const). For two sequences of
equal length the log-probability difference is (s_a - s_b) * logit(q), so each
participant's choice is a logistic function of the switch-count difference
weighted by their own belief strength w_i (logit of their believed switch rate,
scaled by their decision precision).
"""

import numpy as np
import pymc as pm

# Room for participant ids across experiments (ids are unique across a run).
MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def switches(seq):
        seq = seq.strip().upper()
        return float(sum(1 for x, y in zip(seq, seq[1:]) if x != y))

    return {"switch_diff": switches(sequence_a) - switches(sequence_b)}


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population mean and spread of the personal belief weight.
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    w = pm.Deterministic("w", mu_w + sigma_w * z_w)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(w[participant_id] * switch_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
