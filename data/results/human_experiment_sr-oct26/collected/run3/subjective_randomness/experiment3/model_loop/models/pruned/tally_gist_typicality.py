"""Tally-gist typicality.

People keep a running tally of heads minus tails while reading a sequence and
summarise its path by the largest lead either way and the number of returns to
a tie. A sequence looks random to the extent that this tally gist is typical of
their own imagined fair coin (a person-specific believed switch rate), weighed
per flip by a fitted power of length; the final count itself does not matter.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_LEN = 8
N_PEOPLE = 400


def _gist(seq):
    lead, max_lead, ties = 0, 0, 0
    for c in seq:
        lead += 1 if c == "H" else -1
        max_lead = max(max_lead, abs(lead))
        if lead == 0:
            ties += 1
    return max_lead, ties


def _switches(seq):
    return sum(1 for x, y in zip(seq, seq[1:]) if x != y)


# For each length n and tally gist g: how many sequences share g, by switch count.
_TABLE = {}
for _n in range(1, MAX_LEN + 1):
    for _s in itertools.product("HT", repeat=_n):
        _s = "".join(_s)
        key = (_n, _gist(_s))
        row = _TABLE.setdefault(key, np.zeros(MAX_LEN))
        row[_switches(_s)] += 1.0


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        seq = seq.strip().upper()
        counts = _TABLE[(len(seq), _gist(seq))]
        for k in range(MAX_LEN):
            out[f"gist_count_{tag}_{k}"] = float(counts[k])
    out["seq_len"] = float(len(sequence_a.strip()))
    return out


with pm.Model() as model:
    ca = pt.stack(
        [pm.Data(f"gist_count_a_{k}", np.zeros(1)) for k in range(MAX_LEN)], axis=1
    )
    cb = pt.stack(
        [pm.Data(f"gist_count_b_{k}", np.zeros(1)) for k in range(MAX_LEN)], axis=1
    )
    n = pm.Data("seq_len", np.full(1, 4.0))
    pid = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific believed switch rate of a fair coin (non-centred, logit).
    mu_q = pm.Normal("mu_q", 0.5, 0.75)
    sd_q = pm.HalfNormal("sd_q", 1.0)
    z_q = pm.Normal("z_q", 0.0, 1.0, shape=N_PEOPLE)
    q = pm.math.sigmoid(mu_q + sd_q * z_q)[pid][:, None]

    beta = pm.LogNormal("beta", 1.0, 0.75)
    gamma = pm.Normal("gamma", 1.0, 0.5)

    k = pt.arange(MAX_LEN, dtype="float64")[None, :]
    nm1 = (n - 1.0)[:, None]
    valid = pt.le(k, nm1)
    expo_r = pt.switch(valid, nm1 - k, 0.0)
    log_w = k * pt.log(q) + expo_r * pt.log1p(-q)

    def log_gist_prob(c):
        lw = pt.switch(pt.gt(c, 0), pt.log(pt.maximum(c, 1e-12)) + log_w, -1e10)
        return pt.logsumexp(lw, axis=1) + np.log(0.5)

    # Length centred at 5 flips so decisiveness and its length power decouple.
    scale = (n / 5.0) ** gamma
    ra = log_gist_prob(ca) / scale
    rb = log_gist_prob(cb) / scale

    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta * (ra - rb)), 1e-6, 1 - 1e-6)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
