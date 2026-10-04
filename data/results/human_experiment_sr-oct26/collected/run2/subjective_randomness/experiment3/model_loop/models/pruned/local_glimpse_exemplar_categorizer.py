"""Local-glimpse exemplar categorisation of randomness.

People read a sequence through a four-flip window (the whole sequence if
shorter) and compare each glimpse with remembered designed exemplars -- a
streak (HHHH), strict alternation (HTHT) and a two-by-two block (HHTT and its
shifts) -- and with a diffuse memory of random glimpses equally similar to
anything. A glimpse is classed random with probability 1 / (1 + S), S its summed
similarity to the designed exemplars (exponential in mismatched flips; the
alternation and block exemplars share one strength). A sequence looks
random by the share of its glimpses classed random; people pick the sequence
whose glimpses look random more often, with personal decisiveness and a
personal left/right lean.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_SLOTS = 400
WINDOW = 4


def _ham(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _exemplars(period_unit, w):
    seq = (period_unit * 8)
    comp = seq.translate(str.maketrans("HT", "TH"))
    out = set()
    for s in (seq, comp):
        for off in range(len(period_unit)):
            out.add(s[off:off + w])
    return sorted(out)


# Every possible glimpse type (lengths 2, 3, 4) and its distances to exemplars.
TYPES = []
D_STREAK, D_ALT, D_BLOCK, M_BLOCK = [], [], [], []
for _w in (2, 3, 4):
    for _t in itertools.product("HT", repeat=_w):
        _g = "".join(_t)
        TYPES.append(_g)
        D_STREAK.append(min(_ham(_g, e) for e in _exemplars("H", _w)))
        D_ALT.append(min(_ham(_g, e) for e in _exemplars("HT", _w)))
        if _w == 4:
            D_BLOCK.append(min(_ham(_g, e) for e in _exemplars("HHTT", _w)))
            M_BLOCK.append(1.0)
        else:
            D_BLOCK.append(0.0)
            M_BLOCK.append(0.0)
# Glimpses with the same distances to every exemplar are judged alike, so a
# sequence is summarised by its share of glimpses in each distance profile.
PROFILES = sorted(set(zip(D_STREAK, D_ALT, D_BLOCK, M_BLOCK)))
PROFILE_OF = {
    g: PROFILES.index(prof)
    for g, prof in zip(TYPES, zip(D_STREAK, D_ALT, D_BLOCK, M_BLOCK))
}
N_PROFILES = len(PROFILES)
D_STREAK = np.array([p[0] for p in PROFILES], dtype="float64")
D_ALT = np.array([p[1] for p in PROFILES], dtype="float64")
D_BLOCK = np.array([p[2] for p in PROFILES], dtype="float64")
M_BLOCK = np.array([p[3] for p in PROFILES], dtype="float64")


def _glimpse_shares(seq):
    seq = seq.strip().upper()
    w = min(WINDOW, len(seq))
    shares = np.zeros(N_PROFILES)
    n = len(seq) - w + 1
    for i in range(n):
        shares[PROFILE_OF[seq[i:i + w]]] += 1.0 / n
    return shares


def compute_features(sequence_a, sequence_b):
    diff = _glimpse_shares(sequence_a) - _glimpse_shares(sequence_b)
    return {f"glimpse_diff_{i}": float(diff[i]) for i in range(N_PROFILES)}


with pm.Model() as model:
    diffs = [
        pm.Data(f"glimpse_diff_{i}", np.zeros(1, dtype="float64"))
        for i in range(N_PROFILES)
    ]
    diff_mat = pt.stack(diffs, axis=1)  # trials x glimpse profiles
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Exemplar similarity gradient and stored strengths (streak strength = 1).
    # Similarity kept per mismatched flip (similarity = s ** mismatches).
    s_mis = pm.Beta("s_mis", alpha=1.0, beta=5.0)
    # One stored strength for the regular-pattern exemplars (alternation, block).
    log_a_reg = pm.Normal("log_a_reg", mu=-2.0, sigma=1.0)
    c = -pt.log(s_mis)

    sim = (
        pt.exp(-c * D_STREAK)
        + pt.exp(log_a_reg)
        * (pt.exp(-c * D_ALT) + M_BLOCK * pt.exp(-c * D_BLOCK))
    )
    # The diffuse random memory has unit strength, like the streak exemplar.
    p_random_glimpse = pm.Deterministic("p_random_glimpse", 1.0 / (1.0 + sim))

    # Personal decisiveness (non-centred lognormal) and left/right lean.
    mu_beta = pm.Normal("mu_beta", mu=np.log(8.0), sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pt.exp(mu_beta + sigma_beta * z_beta)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    evidence = pt.dot(diff_mat, p_random_glimpse)
    logit = beta[participant_id] * evidence + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
