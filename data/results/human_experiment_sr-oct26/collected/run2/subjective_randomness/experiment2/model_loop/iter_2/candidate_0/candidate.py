"""Random-versus-designed exemplar contrast.

People judge randomness by exemplar memory of both kinds of sequence: remembered
random-looking sequences (balanced H/T, no streak longer than two, no visible
repeating unit) and designed sequences (a short unit repeated at least twice).
A new sequence looks random to the extent it is more similar to the random
exemplars than to the designed ones, similarity falling off exponentially with
the Hamming distance; people choose the sequence with the higher contrast,
differing in how strongly it drives their choice.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_D = 9  # Hamming distances 0..8


def _min_period(s):
    for p in range(1, len(s) + 1):
        if all(s[i] == s[i - p] for i in range(p, len(s))):
            return p
    return len(s)


def _max_run(s):
    best = cur = 1
    for x, y in zip(s, s[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best


_EXEMPLARS = {}


def _exemplars(n):
    if n not in _EXEMPLARS:
        allseq = ["".join(t) for t in itertools.product("HT", repeat=n)]
        designed = [s for s in allseq if _min_period(s) <= n / 2]
        random_ = [
            s for s in allseq
            if _min_period(s) > n / 2
            and abs(s.count("H") - s.count("T")) <= 1
            and _max_run(s) <= 2
        ]
        _EXEMPLARS[n] = (random_, designed)
    return _EXEMPLARS[n]


def _distance_profile(seq, exemplars):
    counts = np.zeros(MAX_D)
    for e in exemplars:
        counts[sum(1 for x, y in zip(seq, e) if x != y)] += 1.0
    return counts / len(exemplars)


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        seq = seq.strip().upper()
        rnd, des = _exemplars(len(seq))
        pr = _distance_profile(seq, rnd)
        pd = _distance_profile(seq, des)
        for d in range(MAX_D):
            out[f"rnd_{tag}_{d}"] = float(pr[d])
            out[f"des_{tag}_{d}"] = float(pd[d])
    return out


with pm.Model() as model:
    def _stack(prefix):
        cols = [pm.Data(f"{prefix}_{d}", np.zeros(1, dtype="float64")) for d in range(MAX_D)]
        return pt.stack(cols, axis=1)

    rnd_a, des_a = _stack("rnd_a"), _stack("des_a")
    rnd_b, des_b = _stack("rnd_b"), _stack("des_b")
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Similarity fall-off per differing flip (shared).
    decay = pm.LogNormal("decay", mu=0.0, sigma=0.7)

    # Person-specific sensitivity, non-centred, spare slots for new people.
    log_beta_mu = pm.Normal("log_beta_mu", mu=0.0, sigma=1.0)
    log_beta_sigma = pm.HalfNormal("log_beta_sigma", sigma=0.5)
    z = pm.Normal("z", 0.0, 1.0, shape=400)
    beta = pt.exp(log_beta_mu + log_beta_sigma * z)

    # Relative weight of resemblance to designed exemplars (shared).
    w_designed = pm.LogNormal("w_designed", mu=0.0, sigma=0.7)

    kernel = pt.exp(-decay * pt.arange(MAX_D))

    def contrast(rnd, des):
        sim_r = pt.dot(rnd, kernel) + 1e-9
        sim_d = pt.dot(des, kernel) + 1e-9
        return pt.log(sim_r) - w_designed * pt.log(sim_d)

    diff = contrast(rnd_a, des_a) - contrast(rnd_b, des_b)
    p = pm.math.sigmoid(beta[participant_id] * diff)
    p_left = pm.Deterministic("p_left", pt.clip(p, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
