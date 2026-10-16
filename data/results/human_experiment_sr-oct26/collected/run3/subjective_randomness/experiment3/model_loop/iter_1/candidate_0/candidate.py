"""Encoding-difficulty (chunk-parse) account of subjective randomness.

People judge randomness by how hard a sequence would be to hold in memory:
they encode it in the most economical way as a short list of chunks, each chunk
either a streak of one face or a stretch of strict alternation, and a sequence
that needs a costlier chunk code looks more random. Holding an alternating chunk
costs a fitted amount relative to holding a streak chunk, so how a sequence is
parsed - and therefore how random it looks - depends on that cost, and each
person acts on the felt encoding difficulty with their own decisiveness and
side habit.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_SLOTS = 400
PAD = 1000.0


def _is_run(seg):
    return all(c == seg[0] for c in seg)


def _is_alt(seg):
    return len(seg) >= 2 and all(x != y for x, y in zip(seg, seg[1:]))


def _parse_counts(seq):
    """All (n_streak_chunks, n_alt_chunks) attainable by parsing seq into chunks."""
    n = len(seq)
    reach = [set() for _ in range(n + 1)]
    reach[0].add((0, 0))
    for i in range(n):
        if not reach[i]:
            continue
        for j in range(i + 1, n + 1):
            seg = seq[i:j]
            kinds = []
            if _is_run(seg):
                kinds.append((1, 0))
            if _is_alt(seg):
                kinds.append((0, 1))
            for dr, da in kinds:
                for r, a in reach[i]:
                    reach[j].add((r + dr, a + da))
    # keep the Pareto front only: a parse dominated in both counts is never minimal
    pts = sorted(reach[n])
    front = [p for p in pts if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in pts)]
    return front


SEQS = ["".join(t) for L in range(1, 9) for t in itertools.product("HT", repeat=L)]
SEQ_INDEX = {s: i for i, s in enumerate(SEQS)}
_fronts = [_parse_counts(s) for s in SEQS]
K = max(len(f) for f in _fronts)
RUN_COUNTS = np.full((len(SEQS), K), PAD)
ALT_COUNTS = np.full((len(SEQS), K), PAD)
for _u, _f in enumerate(_fronts):
    for _k, (_r, _a) in enumerate(_f):
        RUN_COUNTS[_u, _k] = _r
        ALT_COUNTS[_u, _k] = _a
SEQ_LEN = np.array([len(s) for s in SEQS], dtype="float64")


def prepare_observed(rows):
    idx_a, idx_b, pid, y = [], [], [], []
    for r in rows:
        a = str(r["sequence_a"]).strip().upper()
        b = str(r["sequence_b"]).strip().upper()
        idx_a.append(SEQ_INDEX[a])
        idx_b.append(SEQ_INDEX[b])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    return {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    run_counts = pt.as_tensor_variable(RUN_COUNTS)
    alt_counts = pt.as_tensor_variable(ALT_COUNTS)
    seq_len = pt.as_tensor_variable(SEQ_LEN)

    # Cost of holding an alternating chunk, relative to a streak chunk (= 1).
    log_alt_cost = pm.Normal("log_alt_cost", mu=0.0, sigma=1.0)
    alt_cost = pt.exp(log_alt_cost)
    # Encoding difficulty: the cheapest chunk code of each sequence.
    difficulty = pt.min(run_counts + alt_cost * alt_counts, axis=1)
    # Felt difficulty per flip: discounted by a fitted power of the length.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    felt = difficulty / seq_len**gamma

    # Person-specific decisiveness (log-normal, non-centred).
    mu_beta = pm.Normal("mu_beta", mu=0.5, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pt.exp(mu_beta + sigma_beta * z_beta)[participant_id]

    # Person-specific side habit (non-centred).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = (sigma_side * z_side)[participant_id]

    eta = beta * (felt[idx_a] - felt[idx_b]) + side
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(eta), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
