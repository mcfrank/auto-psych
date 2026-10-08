"""Streak-tolerance alarm.

People judge randomness with a streak alarm: there is a tolerance k for how
long a run of identical flips may be before it looks too long to be chance;
every run exceeding k raises the alarm, more so the further it exceeds it (a
smooth hinge), while runs within the tolerance cost nothing. The sequence
raising the smaller alarm is chosen as more random, and people differ in how
strongly the alarm drives their choice (person-level sensitivity).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_RUN = 8
N_SLOTS = 400
HINGE_SHARPNESS = 2.0


def _run_counts(seq):
    seq = seq.strip().upper()
    counts = [0.0] * MAX_RUN
    r = 1
    for k in range(1, len(seq) + 1):
        if k < len(seq) and seq[k] == seq[k - 1]:
            r += 1
        else:
            counts[r - 1] += 1.0
            r = 1
    return counts


def compute_features(sequence_a, sequence_b):
    ca, cb = _run_counts(sequence_a), _run_counts(sequence_b)
    return {f"run_count_diff_{r + 1}": ca[r] - cb[r] for r in range(MAX_RUN)}


RUN_LENGTHS = np.arange(1, MAX_RUN + 1, dtype="float64")

with pm.Model() as model:
    diffs = pt.stack(
        [pm.Data(f"run_count_diff_{r}", np.zeros(1, dtype="float64")) for r in range(1, MAX_RUN + 1)],
        axis=1,
    )  # (n_trials, MAX_RUN)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Streak tolerance, bounded to the possible run lengths 1..MAX_RUN.
    k_logit = pm.Normal("k_logit", mu=0.0, sigma=1.5)
    k = pm.Deterministic("k", 1.0 + (MAX_RUN - 1.0) * pm.math.sigmoid(k_logit))

    # Person-level sensitivity to the alarm (non-centred, log scale).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=1.0)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z)

    excess = pt.softplus(HINGE_SHARPNESS * (RUN_LENGTHS - k)) / HINGE_SHARPNESS
    alarm_diff = pt.dot(diffs, excess)  # alarm(a) - alarm(b)

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(-beta[participant_id] * alarm_diff), 1e-6, 1 - 1e-6)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
