"""Designed-exemplar similarity.

People recognise a designed sequence by its summed similarity to a few
remembered designed exemplars (solid streak, strict alternation, any short
chunk repeated end to end, two solid halves), similarity falling off
exponentially with the number of differing flips (a shared gradient). That
resemblance makes a sequence look made rather than random, by a personal
gain, on top of each person's statistical sense of a random coin (personal
ideal switching rate and sensitivity, lenient toward over-alternation,
personal weights on imbalance and the longest streak, personal guessing).
This single exemplar mechanism replaces the hand-built structural cues.
"""

import functools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_LEN = 8
ND = MAX_LEN + 1


@functools.lru_cache(maxsize=None)
def _designed_exemplars(n):
    pats = set()
    # Any chunk of length p repeated end to end, at least twice (all phases).
    for p in range(1, n // 2 + 1):
        for k in range(2 ** p):
            unit = "".join("H" if (k >> i) & 1 else "T" for i in range(p))
            base = unit * (n // p + 2)
            for phase in range(p):
                pats.add(base[phase : phase + n])
    # Two solid halves.
    h = n // 2
    if h >= 1:
        pats.add("H" * h + "T" * (n - h))
        pats.add("T" * h + "H" * (n - h))
    return tuple(sorted(pats))


def _distance_counts(seq):
    counts = [0.0] * ND
    for e in _designed_exemplars(len(seq)):
        counts[sum(1 for x, y in zip(seq, e) if x != y)] += 1.0
    return counts


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def imbalance(seq):
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    def longest_run_share(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return (best - 1) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    if len(a) != len(b) or not (2 <= len(a) <= MAX_LEN) or set(a + b) - set("HT"):
        raise ValueError(f"unsupported pair: {a!r}, {b!r}")
    out = {
        "seq_len": float(len(a)),
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "maxrun_a": longest_run_share(a),
        "maxrun_b": longest_run_share(b),
    }
    for side, seq in (("a", a), ("b", b)):
        for d, c in enumerate(_distance_counts(seq)):
            out[f"exd{d}_{side}"] = c
    return out


_dist = np.arange(ND, dtype="float64")

with pm.Model() as model:
    seq_len = pm.Data("seq_len", np.full(1, 8.0))
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    maxrun_a = pm.Data("maxrun_a", np.zeros(1, dtype="float64"))
    maxrun_b = pm.Data("maxrun_b", np.zeros(1, dtype="float64"))
    exd_a = pt.stack(
        [pm.Data(f"exd{d}_a", np.ones(1, dtype="float64")) for d in range(ND)], axis=1
    )
    exd_b = pt.stack(
        [pm.Data(f"exd{d}_b", np.ones(1, dtype="float64")) for d in range(ND)], axis=1
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate.
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Personal alternation sensitivity.
    mu_beta = pm.Normal("mu_beta", mu=2.5, sigma=0.5)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta = pm.math.exp(mu_beta + sigma_beta * z_beta)
    # Personal weight on H/T imbalance.
    mu_gamma = pm.Normal("mu_gamma", mu=0.0, sigma=2.0)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.5)
    z_gamma = pm.Normal("z_gamma", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gamma = mu_gamma + sigma_gamma * z_gamma
    # Personal weight on the longest run.
    mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.5)
    z_delta = pm.Normal("z_delta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    delta = mu_delta + sigma_delta * z_delta
    # Leniency toward over-alternation.
    log_over = pm.Normal("log_over", mu=0.0, sigma=1.0)
    over_frac = pm.Deterministic("over_frac", pm.math.exp(log_over))

    # Exemplar similarity: shared gradient per differing flip.
    c_sim = pm.LogNormal("c_sim", mu=np.log(2.0), sigma=0.6)
    # Shared weight on resemblance to designed exemplars (positive: looks made).
    eps = pm.Normal("eps", mu=0.0, sigma=1.5)
    # Personal gain on that resemblance (median 1).
    sigma_gain = pm.HalfNormal("sigma_gain", sigma=0.7)
    z_gain = pm.Normal("z_gain", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gain = pm.math.exp(sigma_gain * z_gain)

    # Personal lapse rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]

    sim_kernel = pt.exp(-c_sim * _dist)  # (ND,)

    def designedness(exd):
        # Log summed similarity to the designed exemplars (counts at each distance).
        return pt.log(pt.sum(exd * sim_kernel[None, :], axis=1) + 1e-3)

    def rate_penalty(rate):
        dev = rate - theta
        return pt.sqr(dev) * pt.switch(dev > 0, over_frac, 1.0)

    b = beta[participant_id]
    g = gamma[participant_id]
    d = delta[participant_id]
    s = gain[participant_id]
    score_a = -b * rate_penalty(alt_rate_a) - g * imbalance_a - d * maxrun_a - s * eps * designedness(exd_a)
    score_b = -b * rate_penalty(alt_rate_b) - g * imbalance_b - d * maxrun_b - s * eps * designedness(exd_b)
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
