"""Structural exemplar categorisation against remembered designed sequences.

People judge randomness by similarity to remembered designs in a structural code: they
remember a few "designed" sequences -- a streak, perfect alternation, and short
repeated patterns (HHTT..., HHT..., HHHT...) -- stored as their rhythm of repeats
(R) and switches (S), so a remembered design matches a new sequence at any phase
and with either starting face. Similarity to each remembered design is
exp(-c * number of the sequence's R/S steps that break it at the best alignment).
A sequence's randomness is minus the log of its summed weighted similarity to
the remembered designs (the less it resembles any design, the more random it
looks); each kind of design (streak, alternation, repeated motif) has its own
fitted memorability, relative to the streak. The more random-looking pair
member is chosen (Luce choice on the two scores), with
person-specific decisiveness and a person-specific side habit at the response
stage.

Exemplars in R/S code (0 = repeat, 1 = switch), matched at every cyclic phase:
  streak "0"; alternation "1"; repeated motifs "01" (HHTT...), "011" (HHT...),
  "001" (HHHTTT...), "0011" (HHHT...), "0001" and "0111" (period-4 rhythms).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

STREAK = ("0",)
ALTERNATION = ("1",)
MOTIFS = ("01", "001", "011", "0001", "0011", "0111")


def _transitions(seq):
    return [1 if x != y else 0 for x, y in zip(seq, seq[1:])]


def _min_mismatch(trans, template):
    """Fewest R/S steps breaking a periodic template at its best phase."""
    p = len(template)
    t = [int(ch) for ch in template]
    best = len(trans)
    for phase in range(p):
        m = sum(1 for i, v in enumerate(trans) if v != t[(i + phase) % p])
        best = min(best, m)
    return best


def _summary(seq):
    trans = _transitions(seq)
    return {
        "streak": float(_min_mismatch(trans, STREAK[0])),
        "alt": float(_min_mismatch(trans, ALTERNATION[0])),
        "motifs": [float(_min_mismatch(trans, m)) for m in MOTIFS],
    }


def prepare_observed(rows):
    """Table of distinct sequences (mismatch counts to each design) plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, pid, y = [], [], [], []
    for r in rows:
        ids = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ids.append(index[seq])
        idx_a.append(ids[0])
        idx_b.append(ids[1])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    return {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "d_streak": np.asarray([t["streak"] for t in table], dtype="float64"),
        "d_alt": np.asarray([t["alt"] for t in table], dtype="float64"),
        "d_motif": np.asarray([t["motifs"] for t in table], dtype="float64").reshape(len(table), len(MOTIFS)),
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    d_streak = pm.Data("d_streak", np.zeros(1, dtype="float64"))
    d_alt = pm.Data("d_alt", np.zeros(1, dtype="float64"))
    d_motif = pm.Data("d_motif", np.zeros((1, len(MOTIFS)), dtype="float64"))

    # Memorability (log weight relative to the streak design) of each kind of design.
    a_alt = pm.Normal("a_alt", mu=-2.0, sigma=1.5)
    a_motif = pm.Normal("a_motif", mu=-3.0, sigma=1.5)
    # Steepness of similarity decay per broken R/S step.
    c = pm.HalfNormal("c", sigma=2.0)

    log_sims = pt.concatenate(
        [
            (-c * d_streak)[:, None],
            (a_alt - c * d_alt)[:, None],
            a_motif - c * d_motif,
        ],
        axis=1,
    )
    log_designed = pm.math.logsumexp(log_sims, axis=1, keepdims=False)
    # Randomness = -log(summed similarity to the remembered designs).
    randomness = -log_designed

    # Person-specific decisiveness (non-centred log-normal population, 400 slots).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Person-specific side habit at the response stage.
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    pid = participant_id
    eta = beta[pid] * (randomness[idx_a] - randomness[idx_b]) + side[pid]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
