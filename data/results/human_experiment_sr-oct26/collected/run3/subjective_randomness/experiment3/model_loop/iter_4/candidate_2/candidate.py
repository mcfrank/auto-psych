"""Take-the-best lexicographic comparison of streak, switching and balance.

People compare the two sequences cue by cue in a fixed order of salience and
let the first cue that clearly tells them apart decide (non-compensatory):
first the longest streak (the shorter longest run looks more random), then,
if the streaks look alike, the number of heads/tails switches, then, if that
too looks alike, heads/tails balance. When no cue separates the pair, the
person's own side habit decides.

Implementation: a cue k with a difference of size |D_k| between the two
sequences is noticed as discriminating with probability
d_k = 1 - exp(-lambda_k |D_k|) (smooth, no hard threshold). Cue k decides the
trial with probability prod_{j<k} (1 - d_j) * d_k; the remaining mass is a
guess. A deciding cue picks the side it favours with log-odds
beta_i * alpha_k (alpha_k: how reliably that cue's direction is followed,
signed; beta_i: a person's heavy-tailed, signed commitment around 1), plus
the person's side habit, which also sets the guess.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _cues(seq):
    seq = seq.strip().upper()
    longest, run = 1, 1
    for x, y in zip(seq, seq[1:]):
        run = run + 1 if x == y else 1
        longest = max(longest, run)
    switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    imbalance = abs(seq.count("H") - seq.count("T"))
    return float(longest), float(switches), float(imbalance)


def compute_features(sequence_a, sequence_b):
    """Signed cue differences, each oriented so that positive favours the left sequence."""
    ra, sa, ia = _cues(sequence_a)
    rb, sb, ib = _cues(sequence_b)
    return {
        "tb_run": rb - ra,  # left has the shorter longest run
        "tb_switch": sa - sb,  # left switches more
        "tb_balance": ib - ia,  # left is better balanced
    }


with pm.Model() as model:
    tb_run = pm.Data("tb_run", np.zeros(1, dtype="float64"))
    tb_switch = pm.Data("tb_switch", np.zeros(1, dtype="float64"))
    tb_balance = pm.Data("tb_balance", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Noticing rates of each cue's difference (per unit of difference).
    log_lam = pm.Normal("log_lam", mu=0.0, sigma=1.0, shape=3)
    lam = pt.exp(log_lam)
    # How reliably each cue's favoured direction is followed (signed log-odds).
    alpha = pm.Normal("alpha", mu=1.0, sigma=1.5, shape=3)

    # Person-specific signed commitment (heavy-tailed, non-centred, around 1).
    sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
    z_beta = pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=400)
    beta = (1.0 + sigma_b * z_beta)[participant_id]

    # Person-specific side habit.
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = (mu_side + sigma_side * z_side)[participant_id]

    diffs = [tb_run, tb_switch, tb_balance]
    remaining = pt.ones_like(tb_run)
    p_mix = pt.zeros_like(tb_run)
    for k, diff in enumerate(diffs):
        d_k = 1.0 - pt.exp(-lam[k] * pt.abs(diff))
        decides = remaining * d_k
        p_mix = p_mix + decides * pm.math.sigmoid(beta * alpha[k] * pt.sgn(diff) + side)
        remaining = remaining * (1.0 - d_k)
    p_mix = p_mix + remaining * pm.math.sigmoid(side)

    p_left = pm.Deterministic("p_left", pt.clip(p_mix, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
