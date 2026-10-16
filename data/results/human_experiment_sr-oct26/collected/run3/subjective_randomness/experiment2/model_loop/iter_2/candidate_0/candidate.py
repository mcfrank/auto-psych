"""Exemplar categorisation of randomness (GCM-style).

People remember examples of "designed" sequences (streaks, perfect
alternation, repeated 3-/4-flip motifs) and of "random-looking" sequences
(balanced heads/tails, no long streak, not perfectly alternating). A sequence
looks random to the extent its summed similarity to the random exemplars
outweighs its summed similarity to the designed exemplars, similarity falling
exponentially with Hamming distance. The pair member with the higher
log similarity ratio is chosen; people differ in decisiveness.
"""
import functools
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_D = 8


def _longest_run(s):
    best = cur = 1
    for x, y in zip(s, s[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best


@functools.lru_cache(maxsize=None)
def _exemplars(length):
    seqs = ["".join(p) for p in itertools.product("HT", repeat=length)]
    designed = []
    for s in seqs:
        for period in (1, 2, 3, 4):
            if length >= 2 * period and all(s[i] == s[i - period] for i in range(period, length)):
                designed.append(s)
                break
    run_tol = max(2, int(round(np.log2(length))))
    alternating = {"".join("HT"[(i + k) % 2] for i in range(length)) for k in (0, 1)}
    rand = []
    for s in seqs:
        if abs(s.count("H") - s.count("T")) > 1:
            continue
        if _longest_run(s) > run_tol:
            continue
        if length >= 4 and s in alternating:
            continue
        rand.append(s)
    return tuple(rand), tuple(designed)


def _hist(seq, exemplars):
    h = [0.0] * (MAX_D + 1)
    for e in exemplars:
        h[sum(1 for x, y in zip(seq, e) if x != y)] += 1.0
    return h


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        seq = seq.strip().upper()
        rand, designed = _exemplars(len(seq))
        hr = _hist(seq, rand)
        hd = _hist(seq, designed)
        for d in range(MAX_D + 1):
            out[f"nr_{tag}_{d}"] = hr[d]
            out[f"nd_{tag}_{d}"] = hd[d]
    return out


with pm.Model() as model:
    cols = {}
    for kind in ("nr", "nd"):
        for tag in ("a", "b"):
            cols[(kind, tag)] = pt.stack(
                [pm.Data(f"{kind}_{tag}_{d}", np.zeros(1, dtype="float64")) for d in range(MAX_D + 1)],
                axis=1,
            )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Similarity steepness: how fast similarity decays per mismatching flip.
    c = pm.LogNormal("c", mu=0.0, sigma=0.75)
    kernel = pt.exp(-c * pt.arange(MAX_D + 1))

    def log_ratio(tag):
        s_r = pt.dot(cols[("nr", tag)], kernel) + 1e-6
        s_d = pt.dot(cols[("nd", tag)], kernel) + 1e-6
        return pt.log(s_r) - pt.log(s_d)

    evidence = log_ratio("a") - log_ratio("b")

    # Person-specific decisiveness (non-centred, log scale).
    mu_beta = pm.Normal("mu_beta", mu=-1.0, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=400)
    beta = pt.exp(mu_beta + sigma_beta * z_beta)

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta[participant_id] * evidence), 1e-6, 1 - 1e-6)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
