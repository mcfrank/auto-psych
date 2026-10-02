"""Exemplar categorisation of random versus designed sequences (GCM-style).

People hold two remembered sets of same-length coin sequences: random-looking
exemplars (head/tail counts within one of balanced, no streak longer than two)
and designed exemplars (solid streaks, strict alternation, repeated pairs
HHTT..., repeated triples HHHTTT..., two halves HHHHTTTT). A sequence looks
random to the extent its summed similarity to the random exemplars outweighs its
summed similarity to the designed ones; similarity falls off exponentially with
the share of flips that differ (Hamming distance / length). The sequence with
the stronger evidence is chosen with a personal decisiveness; people guess on
some trials at a personal lapse rate.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_LEN = 8
LOG_EMPTY = -1000.0


def _longest_run(seq):
    best = cur = 1
    for x, y in zip(seq, seq[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best


def _random_exemplars(n):
    out = []
    for tup in itertools.product("HT", repeat=n):
        s = "".join(tup)
        if abs(s.count("H") - s.count("T")) <= 1 and _longest_run(s) <= 2:
            out.append(s)
    return out


def _designed_exemplars(n):
    out = set()
    for unit in ("H", "HT", "HHTT", "HHHTTT"):
        period = len(unit)
        for shift in range(period):
            rolled = unit[shift:] + unit[:shift]
            s = (rolled * (n // period + 2))[:n]
            out.add(s)
    half = n // 2
    out.add("H" * half + "T" * (n - half))
    out.add("H" * (n - half) + "T" * half)
    flip = str.maketrans("HT", "TH")
    out |= {s.translate(flip) for s in list(out)}
    return sorted(out)


_EXEMPLARS = {
    n: (_random_exemplars(n), _designed_exemplars(n)) for n in range(2, MAX_LEN + 1)
}


def _distance_shares(seq, exemplars):
    counts = np.zeros(MAX_LEN + 1)
    for e in exemplars:
        counts[sum(1 for x, y in zip(seq, e) if x != y)] += 1.0
    shares = counts / len(exemplars)
    # log share of exemplars at each distance; empty bins get a floor far below
    # any reachable similarity so they never contribute.
    return np.where(shares > 0, np.log(np.maximum(shares, 1e-300)), LOG_EMPTY)


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    if len(a) != len(b) or len(a) not in _EXEMPLARS:
        raise ValueError(f"unsupported pair: {a!r}, {b!r}")
    rand_ex, des_ex = _EXEMPLARS[len(a)]
    feats = {"seq_len": float(len(a))}
    for tag, seq in (("a", a), ("b", b)):
        r = _distance_shares(seq, rand_ex)
        d = _distance_shares(seq, des_ex)
        for k in range(MAX_LEN + 1):
            feats[f"logrand_d{k}_{tag}"] = float(r[k])
            feats[f"logdes_d{k}_{tag}"] = float(d[k])
    return feats


with pm.Model() as model:
    seq_len = pm.Data("seq_len", np.full(1, 8.0))
    rand_a = pt.stack(
        [pm.Data(f"logrand_d{k}_a", np.zeros(1)) for k in range(MAX_LEN + 1)], axis=1
    )
    des_a = pt.stack(
        [pm.Data(f"logdes_d{k}_a", np.zeros(1)) for k in range(MAX_LEN + 1)], axis=1
    )
    rand_b = pt.stack(
        [pm.Data(f"logrand_d{k}_b", np.zeros(1)) for k in range(MAX_LEN + 1)], axis=1
    )
    des_b = pt.stack(
        [pm.Data(f"logdes_d{k}_b", np.zeros(1)) for k in range(MAX_LEN + 1)], axis=1
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Shared similarity gradient over the share of mismatched flips: large c means
    # only the nearest exemplars matter, small c means all exemplars count alike.
    c = pm.LogNormal("c", mu=np.log(10.0), sigma=0.8)

    # Personal decisiveness on the random-vs-designed evidence (log-normal population).
    mu_beta = pm.Normal("mu_beta", mu=2.0, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta = pt.exp(mu_beta + sigma_beta * z_beta)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse)

    dist = pt.arange(MAX_LEN + 1, dtype="float64")
    # log similarity kernel per trial and distance: -c * d / n
    log_kernel = -c * dist[None, :] / seq_len[:, None]

    def log_sim(log_shares):
        # log of the mean similarity to an exemplar set
        return pt.logsumexp(log_kernel + log_shares, axis=1)

    # Evidence for "random" over "designed", expressed on the scale of the share
    # of mismatched flips (divided by c): a soft-minimum distance to the designed
    # set minus that to the random set, so c sets only the shape of similarity
    # and the personal decisiveness sets the scale.
    evid_a = (log_sim(rand_a) - log_sim(des_a)) / c
    evid_b = (log_sim(rand_b) - log_sim(des_b)) / c

    b_i = beta[participant_id]
    p_engaged = pm.math.sigmoid(b_i * (evid_a - evid_b))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
