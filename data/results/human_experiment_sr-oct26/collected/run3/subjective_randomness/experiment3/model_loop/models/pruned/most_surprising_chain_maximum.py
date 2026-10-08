"""People judge how random a sequence looks by its single most striking
stretch: the longest unbroken chain of repeats (a streak) or of switches
(strict alternation), whichever is the more improbable for a sequence of that
length under their picture of a fair coin (a fitted believed switch rate).
Only that one most surprising stretch counts; the sequence whose most striking
stretch is less improbable is chosen, with person-specific decisiveness and
side habit.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXM = 7  # sequences up to 8 flips -> up to 7 transitions


def _longest_ones(bits):
    best = cur = 0
    for b in bits:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


# Table of (m transitions, chain length k) cases: row index = m * 8 + k.
# COEF[row, j] counts the transition patterns with j events whose longest
# chain of events is at least k, so P(longest chain >= k) =
# sum_j COEF[row, j] p^j (1-p)^(m-j), p the per-transition chance of the event.
N_CASES = (MAXM + 1) * (MAXM + 1)
COEF = np.zeros((N_CASES, MAXM + 1))
CASE_M = np.zeros(N_CASES)
for _m in range(0, MAXM + 1):
    for _k in range(0, MAXM + 1):
        CASE_M[_m * 8 + _k] = _m
    for _bits in itertools.product((0, 1), repeat=_m):
        L = _longest_ones(_bits)
        for _k in range(0, L + 1):
            COEF[_m * 8 + _k, sum(_bits)] += 1.0


def _chains(seq):
    seq = seq.strip().upper()
    sw = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    rep = [1 - s for s in sw]
    m = len(sw)
    return m * 8 + _longest_ones(rep), m * 8 + _longest_ones(sw)


def compute_features(sequence_a, sequence_b):
    rep_a, sw_a = _chains(sequence_a)
    rep_b, sw_b = _chains(sequence_b)
    return {
        "streak_case_a": float(rep_a),
        "alt_case_a": float(sw_a),
        "streak_case_b": float(rep_b),
        "alt_case_b": float(sw_b),
    }


with pm.Model() as model:
    streak_case_a = pm.Data("streak_case_a", np.zeros(1, dtype="float64"))
    alt_case_a = pm.Data("alt_case_a", np.zeros(1, dtype="float64"))
    streak_case_b = pm.Data("streak_case_b", np.zeros(1, dtype="float64"))
    alt_case_b = pm.Data("alt_case_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Believed per-transition switch rate of a fair coin.
    switch_logit = pm.Normal("switch_logit", mu=0.0, sigma=1.0)
    s = pm.Deterministic("believed_switch", pm.math.sigmoid(switch_logit))

    jj = pt.as_tensor_variable(np.arange(MAXM + 1, dtype="float64"))[None, :]
    mm = pt.as_tensor_variable(CASE_M)[:, None]
    coef = pt.as_tensor_variable(COEF)

    def case_surprise(p):
        basis = pt.exp(jj * pt.log(p) + (mm - jj) * pt.log1p(-p))
        tail = pt.clip(pt.sum(coef * basis, axis=1), 1e-12, 1.0)
        return -pt.log(tail)

    streak_surprise = case_surprise(1.0 - s)  # chains of repeats
    alt_surprise = case_surprise(s)  # chains of switches

    def surprise(streak_case, alt_case):
        # The single most improbable stretch decides.
        return pt.maximum(
            streak_surprise[pt.cast(streak_case, "int64")],
            alt_surprise[pt.cast(alt_case, "int64")],
        )

    diff = surprise(streak_case_b, alt_case_b) - surprise(streak_case_a, alt_case_a)

    # Person-specific decisiveness (log-normal, non-centred) and side habit.
    mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
    sd_beta = pm.HalfNormal("sd_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=400)
    beta = pt.exp(mu_beta + sd_beta * z_beta)
    sd_side = pm.HalfNormal("sd_side", sigma=0.5)
    z_side = pm.Normal("z_side", 0.0, 1.0, shape=400)
    side = sd_side * z_side

    eta = side[participant_id] + beta[participant_id] * diff
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(eta), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
