"""Pattern-coverage detection.

People judge randomness by pattern detection: they scan a sequence for
perceptible regularities -- a streak of identical flips or a stretch of strict
H/T alternation -- and a sequence looks non-random in proportion to how much of
it is covered by a detected pattern. A stretch is noticed once it is long
enough (a soft threshold on its length): streaks at a shared length, alternating
stretches at a length that differs from person to person. What matters is the
share of flips lying inside long runs or long alternations, not the overall
switch rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
RUN_LENGTHS = np.arange(2, 9, dtype="float64")  # run lengths 2..8
ALT_LENGTHS = np.arange(3, 9, dtype="float64")  # alternating-stretch lengths 3..8


def _coverage(seq):
    """Share of flips inside a run / alternating stretch of each length."""
    seq = seq.strip().upper()
    n = len(seq)
    if n < 2:
        raise ValueError(f"sequence too short: {seq!r}")
    run_len = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        for k in range(i, j + 1):
            run_len[k] = j - i + 1
        i = j + 1
    alt_len = [1] * n
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n and seq[j + 1] != seq[j]:
            j += 1
        if j > i:
            for k in range(i, j + 1):
                alt_len[k] = max(alt_len[k], j - i + 1)
            i = j
        else:
            i += 1
    run_cov = [sum(1 for r in run_len if r == L) / n for L in RUN_LENGTHS]
    alt_cov = [sum(1 for a in alt_len if a == L) / n for L in ALT_LENGTHS]
    return run_cov, alt_cov


def compute_features(sequence_a, sequence_b):
    run_a, alt_a = _coverage(sequence_a)
    run_b, alt_b = _coverage(sequence_b)
    out = {}
    for L, x, y in zip(RUN_LENGTHS, run_a, run_b):
        out[f"drun_{int(L)}"] = x - y
    for L, x, y in zip(ALT_LENGTHS, alt_a, alt_b):
        out[f"dalt_{int(L)}"] = x - y
    return out


with pm.Model() as model:
    drun = pt.stack(
        [pm.Data(f"drun_{int(L)}", np.zeros(1, dtype="float64")) for L in RUN_LENGTHS],
        axis=1,
    )
    dalt = pt.stack(
        [pm.Data(f"dalt_{int(L)}", np.zeros(1, dtype="float64")) for L in ALT_LENGTHS],
        axis=1,
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Detection sharpness and pattern sensitivity.
    sharp = pm.LogNormal("sharp", mu=0.3, sigma=0.5)
    beta = pm.LogNormal("beta", mu=1.5, sigma=0.7)
    # Shared streak-detection threshold (in flips).
    k_run = pm.Normal("k_run", mu=3.0, sigma=1.5)
    # Personal alternation-detection threshold (non-centred).
    mu_alt = pm.Normal("mu_alt", mu=5.0, sigma=2.0)
    sigma_alt = pm.HalfNormal("sigma_alt", sigma=1.5)
    z_alt = pm.Normal("z_alt", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    k_alt = pm.Deterministic("k_alt", mu_alt + sigma_alt * z_alt)

    det_run = pm.math.sigmoid(sharp * (pt.as_tensor(RUN_LENGTHS) - k_run))  # (7,)
    det_alt = pm.math.sigmoid(
        sharp * (pt.as_tensor(ALT_LENGTHS)[None, :] - k_alt[participant_id][:, None])
    )  # (N, 6)
    # Pattern coverage of A minus that of B.
    d_cov = pt.dot(drun, det_run) + pt.sum(dalt * det_alt, axis=1)
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(-beta * 4.0 * d_cov), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
