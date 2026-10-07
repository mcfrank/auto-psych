"""Designed-exemplar similarity (GCM-style exemplar account of perceived randomness).

People carry a few remembered examples of obviously "designed" coin sequences --
streaks (HHHH..., TTTT...), perfect alternation (HTHT..., THTH...) and short
repeated motifs (HHT, HTT, HHTT repeated) -- and a sequence looks random to the
extent it is dissimilar from all of them. Similarity to an exemplar decays
exponentially with the Hamming distance (number of mismatching flips), summed
over the exemplars of each kind (so the nearest dominate). The sequence with the
smaller designed-exemplar similarity is chosen as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_D = 8
CLASSES = ("const", "alt", "motif")
MOTIFS = ("HHT", "HTT", "HHTT")


def _exemplars(n):
    def tile(m):
        return (m * (n // len(m) + 2))[:n]

    const = {"H" * n, "T" * n}
    alt = {tile("HT"), tile("TH")}
    motif = set()
    for m in MOTIFS:
        if n >= len(m) + 2:
            for s in range(len(m)):
                motif.add(tile(m[s:] + m[:s]))
    motif -= const | alt
    return {"const": sorted(const), "alt": sorted(alt), "motif": sorted(motif)}


def _dist_shares(seq):
    """For each class, the share of its exemplars at each Hamming distance 0..MAX_D."""
    seq = seq.strip().upper()
    ex = _exemplars(len(seq))
    out = {}
    for c in CLASSES:
        hist = np.zeros(MAX_D + 1)
        for e in ex[c]:
            hist[sum(1 for x, y in zip(seq, e) if x != y)] += 1.0
        if ex[c]:
            hist /= len(ex[c])
        out[c] = hist
    return out


def compute_features(sequence_a, sequence_b):
    feats = {}
    for side, seq in (("a", sequence_a), ("b", sequence_b)):
        shares = _dist_shares(seq)
        for c in CLASSES:
            for d in range(MAX_D + 1):
                feats[f"{c}_d{d}_{side}"] = float(shares[c][d])
    return feats


with pm.Model() as model:
    data = {}
    for side in ("a", "b"):
        for c in CLASSES:
            cols = [
                pm.Data(f"{c}_d{d}_{side}", np.zeros(1, dtype="float64"))
                for d in range(MAX_D + 1)
            ]
            data[(c, side)] = pt.stack(cols, axis=1)  # (trials, MAX_D+1)

    # How steeply similarity falls per mismatching flip.
    specificity = pm.LogNormal("specificity", mu=0.0, sigma=0.75)
    # Memory strength of alternation and motif exemplars relative to streaks,
    # and of the diffuse background (anything-goes) against which similarity is judged.
    log_w_alt = pm.Normal("log_w_alt", mu=0.0, sigma=1.5)
    log_w_motif = pm.Normal("log_w_motif", mu=0.0, sigma=1.5)
    log_background = pm.Normal("log_background", mu=-1.0, sigma=1.5)
    beta = pm.HalfNormal("beta", sigma=3.0)

    decay = pt.exp(-specificity * pt.arange(MAX_D + 1, dtype="float64"))
    weights = {"const": 1.0, "alt": pt.exp(log_w_alt), "motif": pt.exp(log_w_motif)}

    def randomness(side):
        sim = sum(weights[c] * pt.dot(data[(c, side)], decay) for c in CLASSES)
        return -pt.log(pt.exp(log_background) + sim)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (randomness("a") - randomness("b")))
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
