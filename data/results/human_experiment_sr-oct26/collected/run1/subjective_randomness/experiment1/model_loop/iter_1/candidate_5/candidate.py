"""Run-hazard switch belief (refinement of personal_switch_belief).

Each person judges as more random the sequence that is more probable under
their own subjective model of a coin as a switching process. The one change
from personal_switch_belief: the believed coin carries the gambler's fallacy —
its switch probability grows with the length of the current run,
    P(switch | run length r) = sigmoid(a_i + b * (r - 1)),
with a personal baseline a_i and a shared growth b — so long runs are judged
less probable than their switch count alone implies.

The log-probability difference between two equal-length sequences is
    sum_r ds_r * log sigmoid(h_r) + dc_r * log sigmoid(-h_r),
where ds_r / dc_r are the differences (left minus right) in how many times a
run of length r ended in a switch / continued; p_left = sigmoid(that).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_RUN = 7  # sequences have at most 8 flips, so a run before a flip is <= 7


def _run_counts(seq):
    seq = seq.strip().upper()
    if len(seq) < 2:
        raise ValueError(f"sequence too short: {seq!r}")
    s = [0.0] * MAX_RUN
    c = [0.0] * MAX_RUN
    run = 1
    for x, y in zip(seq, seq[1:]):
        if x != y:
            s[run - 1] += 1.0
            run = 1
        else:
            c[run - 1] += 1.0
            run += 1
    return s, c


def compute_features(sequence_a, sequence_b):
    sa, ca = _run_counts(sequence_a)
    sb, cb = _run_counts(sequence_b)
    out = {}
    for r in range(MAX_RUN):
        out[f"ds_{r + 1}"] = sa[r] - sb[r]
        out[f"dc_{r + 1}"] = ca[r] - cb[r]
    return out


with pm.Model() as model:
    ds = pt.stack(
        [pm.Data(f"ds_{r + 1}", np.zeros(1, dtype="float64")) for r in range(MAX_RUN)],
        axis=1,
    )
    dc = pt.stack(
        [pm.Data(f"dc_{r + 1}", np.zeros(1, dtype="float64")) for r in range(MAX_RUN)],
        axis=1,
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal baseline switch belief (logit scale), non-centred.
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    a = mu_a + sigma_a * z_a
    # Shared growth of the believed switch probability with run length.
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    r_idx = pt.arange(MAX_RUN).astype("float64")
    # Per-participant believed log P(switch) / log P(continue) by run length.
    h = a[:, None] + b * r_idx[None, :]  # participants x run length
    log_switch = -pt.softplus(-h)
    log_stay = -pt.softplus(h)
    dlogp = pt.sum(
        ds * log_switch[participant_id] + dc * log_stay[participant_id], axis=1
    )

    # Smoothly bound the log-odds (linear over any realistic range) so p_left
    # stays inside (0, 1) without a hard clip that would flatten gradients.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(30.0 * pt.tanh(dlogp / 30.0)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
