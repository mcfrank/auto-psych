"""Exemplar account of subjective randomness.

People hold remembered exemplars of designed coin sequences (streaks, strict
alternation, short repeating units and their phases/mirrors) and judge a
sequence non-random by its summed similarity to them (similarity decays
exponentially with the Hamming distance to an exemplar, as a share of the
length; generalized context model). Random exemplars are diffuse and favour no
sequence, so the choice goes to the sequence less similar to the designed
exemplars. People differ in how strongly the alternation exemplars are stored.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_LEN = 8
N_BINS = MAX_LEN + 1  # Hamming distances 0..8
ALT_UNITS = ["HT"]
OTHER_UNITS = ["H", "HHT", "HHTT", "HHHT", "HTTT"]


def _exemplars(units, n):
    out = set()
    for u in units:
        for comp in (False, True):
            unit = u.translate(str.maketrans("HT", "TH")) if comp else u
            for shift in range(len(unit)):
                s = unit[shift:] + unit[:shift]
                out.add((s * (n // len(s) + 2))[:n])
    return sorted(out)


def _hist(seq, units):
    n = len(seq)
    counts = np.zeros(N_BINS)
    for ex in _exemplars(units, n):
        counts[sum(a != b for a, b in zip(seq, ex))] += 1.0
    return counts


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    feats = {"seq_len": float(len(a))}
    for tag, seq in (("a", a), ("b", b)):
        for fam, units in (("alt", ALT_UNITS), ("oth", OTHER_UNITS)):
            h = _hist(seq, units)
            for k in range(N_BINS):
                feats[f"{fam}_{tag}_{k}"] = float(h[k])
    return feats


with pm.Model() as model:
    seq_len = pm.Data("seq_len", np.ones(1, dtype="float64"))
    cols = {}
    for tag in ("a", "b"):
        for fam in ("alt", "oth"):
            cols[(fam, tag)] = pt.stack(
                [pm.Data(f"{fam}_{tag}_{k}", np.zeros(1, dtype="float64")) for k in range(N_BINS)],
                axis=1,
            )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Similarity gradient: how fast similarity decays with share of flips changed.
    log_c = pm.Normal("log_c", mu=np.log(6.0), sigma=0.7)
    c = pm.Deterministic("c", pt.exp(log_c))
    # Decision sensitivity to the difference in log summed similarity.
    beta = pm.HalfNormal("beta", sigma=2.0)
    # Person-specific storage strength (log weight) of the alternation exemplars.
    mu_alt = pm.Normal("mu_alt", mu=0.0, sigma=1.5)
    sigma_alt = pm.HalfNormal("sigma_alt", sigma=1.5)
    z_alt = pm.Normal("z_alt", 0.0, 1.0, shape=400)
    w_alt = mu_alt + sigma_alt * z_alt[participant_id]

    dist = pt.arange(N_BINS, dtype="float64")[None, :] / seq_len[:, None]
    decay = pt.exp(-c * dist)

    def log_sim(tag):
        s_alt = pt.sum(cols[("alt", tag)] * decay, axis=1)
        s_oth = pt.sum(cols[("oth", tag)] * decay, axis=1)
        return pt.log(s_oth + pt.exp(w_alt) * s_alt + 1e-12)

    designed_a = log_sim("a")
    designed_b = log_sim("b")
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta * (designed_b - designed_a)), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
