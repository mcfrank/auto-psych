"""Bayesian evidence for a coin over a regular-rhythm generator, with a personal coin.

People judge randomness by Bayesian model comparison: a sequence looks random to
the extent that a coin explains it better than a "regular-rhythm" generator that
builds sequences from runs of a few recurring lengths (run lengths drawn i.i.d.
from an unknown categorical distribution with a symmetric Dirichlet prior), so
sequences whose runs all share one length look designed and sequences with a
variety of run lengths look random. The one distortion: each person's mental
model of the fair coin switches at their own personal rate rather than 1/2.
People also guess on some trials at a personal lapse rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_LEN = 8


def _clean(seq):
    seq = str(seq).strip().upper()
    if len(seq) < 2 or len(seq) > MAX_LEN or set(seq) - {"H", "T"}:
        raise ValueError(f"unsupported sequence: {seq!r}")
    return seq


def _run_lengths(seq):
    out = []
    cur = 1
    for x, y in zip(seq, seq[1:]):
        if x == y:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


def prepare_observed(rows):
    """Distinct-sequence table plus per-trial indices; dummy responses when absent."""
    rows = list(rows)
    if not rows:
        raise ValueError("prepare_observed requires at least one row.")
    seqs_a = [_clean(r["sequence_a"]) for r in rows]
    seqs_b = [_clean(r["sequence_b"]) for r in rows]
    table = sorted(set(seqs_a) | set(seqs_b))
    index = {s: i for i, s in enumerate(table)}
    counts = np.zeros((len(table), MAX_LEN), dtype="float64")
    n = np.zeros(len(table), dtype="float64")
    switches = np.zeros(len(table), dtype="float64")
    for i, s in enumerate(table):
        rl = _run_lengths(s)
        for length in rl:
            counts[i, length - 1] += 1.0
        n[i] = len(s)
        switches[i] = len(rl) - 1
    return {
        "seq_runlen_counts": counts,
        "seq_len": n,
        "seq_switches": switches,
        "seq_repeats": n - 1.0 - switches,
        "idx_a": np.array([index[s] for s in seqs_a], dtype="int64"),
        "idx_b": np.array([index[s] for s in seqs_b], dtype="int64"),
        "participant_id": np.array([int(float(r["participant_id"])) for r in rows], dtype="int64"),
        "chose_left": np.array(
            [int(float(r["chose_left"])) if "chose_left" in r else 0 for r in rows], dtype="int64"
        ),
    }


with pm.Model() as model:
    seq_runlen_counts = pm.Data("seq_runlen_counts", np.zeros((1, MAX_LEN), dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_switches = pm.Data("seq_switches", np.zeros(1, dtype="float64"))
    seq_repeats = pm.Data("seq_repeats", np.zeros(1, dtype="float64"))
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal believed switch rate of a fair coin (non-centred logit-normal).
    mu_theta = pm.Normal("mu_theta", mu=0.3, sigma=1.0)
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
    z_theta = pm.Normal("z_theta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu_theta + sigma_theta * z_theta))

    # Concentration of the regular-rhythm generator's Dirichlet prior over run
    # lengths (small: it expects runs of one recurring length).
    log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=1.0)
    alpha = pm.math.exp(log_alpha)

    # Personal decisiveness on the log evidence (non-centred log-normal).
    mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta = pm.math.exp(mu_beta + sigma_beta * z_beta)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    # Dirichlet-multinomial log marginal of each distinct sequence's run lengths
    # (K = sequence length possible run lengths), computed once per sequence.
    n_runs = pt.sum(seq_runlen_counts, axis=1)
    k_alpha = seq_len * alpha
    log_rhythm = (
        pt.gammaln(k_alpha)
        - pt.gammaln(k_alpha + n_runs)
        + pt.sum(pt.gammaln(alpha + seq_runlen_counts) - pt.gammaln(alpha), axis=1)
    )

    th = theta[participant_id]
    log_th = pt.log(th)
    log_1mth = pt.log1p(-th)

    # Log evidence for the (personal) coin over the rhythm generator; the first
    # flip is 1/2 under both and cancels.
    ev_a = seq_switches[idx_a] * log_th + seq_repeats[idx_a] * log_1mth - log_rhythm[idx_a]
    ev_b = seq_switches[idx_b] * log_th + seq_repeats[idx_b] * log_1mth - log_rhythm[idx_b]

    p_engaged = pm.math.sigmoid(beta[participant_id] * (ev_a - ev_b))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
