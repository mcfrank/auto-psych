"""Exemplar judgment in a gist space: remembered coin outcomes versus remembered designs.

People judge randomness by exemplar similarity in a psychological space of
sequence gist — switch rate, heads/tails imbalance and longest-streak length,
each scaled to [0, 1] for the sequence's length. They compare a new sequence
with remembered examples of what a real coin produces (all 2**n sequences of
that length, each remembered in proportion to its probability under their own
picture of a coin that switches with probability q) and with a few remembered
"designed" sequences of that length (a streak, perfect alternation, repeated
HHT and repeated HHTT, each with its own memorability). Similarity is
exp(-c * attention-weighted city-block distance) (GCM). A sequence's
randomness is log(summed similarity to random exemplars) - log(summed
similarity to designed exemplars); the choice is a logistic function of the
randomness difference, scaled by a person's signed sensitivity (Normal population, can cross zero),
plus a person's side habit.

Random exemplars sharing the same gist point are pooled (log count), so each
sequence has at most a few dozen random exemplar points. Unique sequences are
tabulated once in prepare_observed; trials index into the table.
"""

import itertools
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_PEOPLE = 400
DESIGN_MOTIFS = ("H", "HT", "HHT", "HHTT")


def _gist(seq):
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    longest, run = 1, 1
    for x, y in zip(seq, seq[1:]):
        run = run + 1 if x == y else 1
        longest = max(longest, run)
    return k, (k / (n - 1), abs(2 * h - n) / n, (longest - 1) / (n - 1))


def _random_points(n):
    pooled = {}
    for tup in itertools.product("HT", repeat=n):
        k, g = _gist("".join(tup))
        key = (k, g)
        pooled[key] = pooled.get(key, 0) + 1
    return [(k, g, math.log(c)) for (k, g), c in sorted(pooled.items())]


_RANDOM = {n: _random_points(n) for n in range(2, 9)}
J_MAX = max(len(v) for v in _RANDOM.values())
D_N = len(DESIGN_MOTIFS)


def _design_gists(n):
    return [_gist((m * n)[:n])[1] for m in DESIGN_MOTIFS]


def _tables(seq):
    seq = seq.strip().upper()
    n = len(seq)
    _, g = _gist(seq)
    pts = _RANDOM[n]
    dist = np.zeros((3, J_MAX))
    kk = np.zeros(J_MAX)
    rr = np.zeros(J_MAX)
    lc = np.full(J_MAX, -1e4)
    for j, (k, gj, logc) in enumerate(pts):
        dist[:, j] = [abs(g[d] - gj[d]) for d in range(3)]
        kk[j] = k
        rr[j] = n - 1 - k
        lc[j] = logc
    ddist = np.zeros((3, D_N))
    for t, gd in enumerate(_design_gists(n)):
        ddist[:, t] = [abs(g[d] - gd[d]) for d in range(3)]
    return dist, kk, rr, lc, ddist


def prepare_observed(rows):
    seqs = {}
    ia, ib = [], []
    for r in rows:
        for s, idx in ((r["sequence_a"], ia), (r["sequence_b"], ib)):
            s = s.strip().upper()
            if s not in seqs:
                seqs[s] = len(seqs)
            idx.append(seqs[s])
    tabs = [_tables(s) for s in seqs]
    out = {
        "seq_a_idx": np.array(ia, dtype="int64"),
        "seq_b_idx": np.array(ib, dtype="int64"),
        "rdist": np.stack([t[0] for t in tabs]).reshape(len(tabs), 3 * J_MAX),
        "rk": np.stack([t[1] for t in tabs]),
        "rr": np.stack([t[2] for t in tabs]),
        "rlogc": np.stack([t[3] for t in tabs]),
        "ddist": np.stack([t[4] for t in tabs]).reshape(len(tabs), 3 * D_N),
        "participant_id": np.array([int(float(r["participant_id"])) for r in rows], dtype="int64"),
        "chose_left": np.array([int(float(r.get("chose_left", 0) or 0)) for r in rows], dtype="int64"),
    }
    return out


with pm.Model() as model:
    seq_a_idx = pm.Data("seq_a_idx", np.zeros(1, dtype="int64"))
    seq_b_idx = pm.Data("seq_b_idx", np.zeros(1, dtype="int64"))
    rdist = pm.Data("rdist", np.zeros((1, 3 * J_MAX), dtype="float64"))
    rk = pm.Data("rk", np.zeros((1, J_MAX), dtype="float64"))
    rr = pm.Data("rr", np.zeros((1, J_MAX), dtype="float64"))
    rlogc = pm.Data("rlogc", np.zeros((1, J_MAX), dtype="float64"))
    ddist = pm.Data("ddist", np.zeros((1, 3 * D_N), dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # GCM similarity: specificity c and attention over the three gist dimensions.
    c = pm.LogNormal("c", mu=math.log(4.0), sigma=0.7)
    attn = pm.Dirichlet("attention", a=np.full(3, 2.0))
    # Remembered coin's switch probability (weights the random exemplars).
    q = pm.Beta("q", alpha=4.0, beta=4.0)
    # Relative memorability of the designed exemplars (streak fixed at 0).
    mem_free = pm.Normal("mem_free", mu=0.0, sigma=1.0, shape=D_N - 1)
    log_mem = pt.concatenate([pt.zeros(1), mem_free])

    rd = rdist.reshape((rdist.shape[0], 3, J_MAX))
    dd = ddist.reshape((ddist.shape[0], 3, D_N))
    r_wdist = pt.sum(attn[None, :, None] * rd, axis=1)
    d_wdist = pt.sum(attn[None, :, None] * dd, axis=1)

    log_w = rlogc + rk * pt.log(q) + rr * pt.log1p(-q)
    log_random = pm.math.logsumexp(log_w - c * r_wdist, axis=1).flatten()
    log_design = pm.math.logsumexp(log_mem[None, :] - c * d_wdist, axis=1).flatten()
    randomness = log_random - log_design

    # Person-level decisiveness (signed, non-centred) and side habit.
    mu_b = pm.Normal("mu_b", mu=1.0, sigma=1.0)
    sd_b = pm.HalfNormal("sd_b", sigma=0.5)
    z_b = pm.Normal("z_b", mu=0.0, sigma=1.0, shape=N_PEOPLE)
    beta_i = mu_b + sd_b * z_b
    mu_s = pm.Normal("mu_s", mu=0.0, sigma=0.5)
    sd_s = pm.HalfNormal("sd_s", sigma=0.5)
    z_s = pm.Normal("z_s", mu=0.0, sigma=1.0, shape=N_PEOPLE)
    side_i = mu_s + sd_s * z_s

    diff = randomness[seq_a_idx] - randomness[seq_b_idx]
    eta = beta_i[participant_id] * diff + side_i[participant_id]
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(eta), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
