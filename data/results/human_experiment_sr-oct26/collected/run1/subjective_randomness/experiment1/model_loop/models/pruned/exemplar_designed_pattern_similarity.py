"""Exemplar similarity to remembered designed patterns.

People carry a few remembered examples of obviously designed coin sequences
(a solid streak, strict alternation, pairs, triples, two halves) and a sequence
looks non-random to the extent that it closely resembles any of them, with
similarity falling off exponentially with the share of mismatched flips
(Hamming distance to the nearest phase/complement of each exemplar, divided by
length). People differ in how prominently strict alternation is among their
remembered designed exemplars; the other exemplars are shared.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
FAMILIES = ["streak", "alt", "pairs", "triples", "halves"]


def _periodic(block, length):
    """All phase shifts and complements of a repeating H/T block pattern."""
    period = 2 * block
    base = ("H" * block + "T" * block) * (length // period + 2)
    out = set()
    for shift in range(period):
        out.add(base[shift:shift + length])
    return out


def _templates(family, length):
    if family == "streak":
        return {"H" * length, "T" * length}
    if family == "alt":
        return _periodic(1, length)
    if family == "pairs":
        return _periodic(2, length)
    if family == "triples":
        return _periodic(3, length)
    if family == "halves":
        out = set()
        for k in {length // 2, (length + 1) // 2}:
            out.add("H" * k + "T" * (length - k))
            out.add("T" * k + "H" * (length - k))
        return out
    raise ValueError(family)


def _min_distance(seq, family):
    seq = seq.strip().upper()
    n = len(seq)
    if n < 2:
        raise ValueError(f"sequence too short: {seq!r}")
    return min(sum(a != b for a, b in zip(seq, t)) for t in _templates(family, n)) / n


def compute_features(sequence_a, sequence_b):
    feats = {}
    for fam in FAMILIES:
        feats[f"d_{fam}_a"] = float(_min_distance(sequence_a, fam))
        feats[f"d_{fam}_b"] = float(_min_distance(sequence_b, fam))
    return feats


with pm.Model() as model:
    d_a = {f: pm.Data(f"d_{f}_a", np.zeros(1, dtype="float64")) for f in FAMILIES}
    d_b = {f: pm.Data(f"d_{f}_b", np.zeros(1, dtype="float64")) for f in FAMILIES}
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # How sharply similarity falls off with the share of mismatched flips.
    lam = pm.LogNormal("lam", mu=np.log(8.0), sigma=0.5)
    # Decision sensitivity to the difference in log designed-similarity.
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)
    # Log prominence of the shared non-alternation exemplars (streak fixed at 0).
    log_w_shared = pm.Normal("log_w_shared", mu=0.0, sigma=1.0, shape=3)
    # Personal log prominence of the strict-alternation exemplar (non-centred).
    mu_alt = pm.Normal("mu_alt", mu=0.0, sigma=1.5)
    sigma_alt = pm.HalfNormal("sigma_alt", sigma=1.5)
    z_alt = pm.Normal("z_alt", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    log_w_alt = mu_alt + sigma_alt * z_alt[participant_id]

    def log_designed(d):
        terms = pt.stack(
            [
                -lam * d["streak"],
                log_w_alt - lam * d["alt"],
                log_w_shared[0] - lam * d["pairs"],
                log_w_shared[1] - lam * d["triples"],
                log_w_shared[2] - lam * d["halves"],
            ],
            axis=0,
        )
        return pm.math.logsumexp(terms, axis=0).flatten()

    # Left chosen when it resembles designed exemplars less than the right does.
    p_left = pm.Deterministic(
        "p_left",
        pt.clip(pm.math.sigmoid(beta * (log_designed(d_b) - log_designed(d_a))), 1e-6, 1 - 1e-6),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
