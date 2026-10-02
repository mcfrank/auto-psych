"""Coin-belief exemplar cloud: random-vs-designed similarity categorisation.

People judge randomness by exemplar categorisation. Their "random" category is
a remembered cloud of every same-length coin sequence, each as familiar as it is
probable under a shared belief about how often a fair coin switches sides; their
"designed" category is a handful of rigid patterns (solid streak, strict
alternation, repeated pairs, repeated triples, two halves). A sequence looks
random to the extent its summed similarity to the random cloud outweighs its
summed similarity to the designed patterns, similarity falling off exponentially
with the share of flips that differ. People differ only in how decisively they
act on this evidence.
"""

import functools
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_LEN = 8
ND = MAX_LEN + 1  # Hamming distances 0..8
NK = MAX_LEN  # switch counts 0..7


@functools.lru_cache(maxsize=None)
def _all_sequences(n):
    return ["".join(s) for s in itertools.product("HT", repeat=n)]


def _switches(seq):
    return sum(1 for x, y in zip(seq, seq[1:]) if x != y)


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


@functools.lru_cache(maxsize=None)
def _designed(n):
    pats = set()
    for block in (1, 2, 3):
        for start in "HT":
            other = "T" if start == "H" else "H"
            unit = start * block + other * block
            base = (unit * n)[: 2 * n]
            for phase in range(2 * block):
                pats.add(base[phase : phase + n])
    for c in "HT":
        pats.add(c * n)
    h = n // 2
    if h >= 1:
        pats.add("H" * h + "T" * (n - h))
        pats.add("T" * h + "H" * (n - h))
    return tuple(sorted(pats))


@functools.lru_cache(maxsize=None)
def _seq_counts(seq):
    n = len(seq)
    r = np.zeros((ND, NK))
    for s in _all_sequences(n):
        r[_hamming(seq, s), _switches(s)] += 1.0
    d = np.zeros(ND)
    for s in _designed(n):
        d[_hamming(seq, s)] += 1.0
    return r, d


def _code(seq):
    # Unique index for every H/T sequence of length 1..MAX_LEN.
    return (2 ** len(seq) - 2) + int(seq.replace("H", "0").replace("T", "1"), 2)


def _build_tables():
    n_codes = 2 ** (MAX_LEN + 1) - 2
    r_tab = np.zeros((n_codes, ND, NK))
    d_tab = np.zeros((n_codes, ND))
    for n in range(1, MAX_LEN + 1):
        for seq in _all_sequences(n):
            r, d = _seq_counts(seq)
            r_tab[_code(seq)] = r
            d_tab[_code(seq)] = d
    code_len = np.ones(n_codes)
    for n in range(1, MAX_LEN + 1):
        code_len[2 ** n - 2 : 2 ** (n + 1) - 2] = n
    return r_tab, d_tab, code_len


R_TABLE, D_TABLE, CODE_LEN = _build_tables()


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    if len(a) != len(b) or not (2 <= len(a) <= MAX_LEN) or set(a + b) - set("HT"):
        raise ValueError(f"unsupported pair: {a!r}, {b!r}")
    return {"seq_len": float(len(a)), "code_a": _code(a), "code_b": _code(b)}


_dist = np.arange(ND, dtype="float64")
_sw = np.arange(NK, dtype="float64")

with pm.Model() as model:
    seq_len = pm.Data("seq_len", np.full(1, 8.0))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    code_a = pm.Data("code_a", np.zeros(1, dtype="int64"))
    code_b = pm.Data("code_b", np.zeros(1, dtype="int64"))

    # Shared belief about a fair coin's switch rate (logit scale).
    logit_q = pm.Normal("logit_q", mu=0.3, sigma=0.7)
    q = pm.Deterministic("q", pm.math.sigmoid(logit_q))

    # Shared similarity gradient (per share of mismatched flips).
    c = pm.LogNormal("c", mu=np.log(20.0), sigma=1.0)

    # Personal decisiveness on the random-vs-designed evidence (non-centred).
    mu_tau = pm.Normal("mu_tau", mu=0.0, sigma=1.0)
    sigma_tau = pm.HalfNormal("sigma_tau", sigma=0.5)
    z_tau = pm.Normal("z_tau", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    tau_all = pm.math.exp(mu_tau + sigma_tau * z_tau)

    # Shared guessing rate.
    lam = pm.Beta("lapse", alpha=1.5, beta=10.0)

    # Similarity of every sequence to each distance class, summed once per code:
    # A[code, k] = sum_d R[code, d, k] exp(-c d / L_code), likewise for designed.
    s_code = pt.exp(-c * _dist[None, :] / pt.constant(CODE_LEN)[:, None])  # (codes, ND)
    a_tab = pt.sum(pt.constant(R_TABLE) * s_code[:, :, None], axis=1)  # (codes, NK)
    des_tab = pt.sum(pt.constant(D_TABLE) * s_code, axis=1)  # (codes,)

    L = seq_len
    # Familiarity of a cloud member with k switches under the person's coin belief.
    w = pt.exp(
        np.log(0.5)
        + _sw[None, :] * pt.log(q)
        + (L[:, None] - 1.0 - _sw[None, :]) * pt.log1p(-q)
    )  # (n, NK)

    def evidence(code):
        s_rand = pt.sum(a_tab[code] * w, axis=1)
        return pt.log(s_rand) - pt.log(des_tab[code])

    diff = evidence(code_a) - evidence(code_b)
    p_engaged = pm.math.sigmoid(tau_all[participant_id] * diff)
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
