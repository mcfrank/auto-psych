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
guess. A deciding cue picks the side it favours with one shared log-odds of
commitment (scaled by a person's own positive decisiveness), plus the
person's side habit, which alone sets the guess.
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
    """Cue difference sizes and the side each cue favours (+1 left, -1 right, 0 tie)."""
    ra, sa, ia = _cues(sequence_a)
    rb, sb, ib = _cues(sequence_b)
    d_run, d_sw, d_bal = rb - ra, sa - sb, ib - ia
    return {
        "tb_run_size": abs(d_run),
        "tb_run_dir": float(np.sign(d_run)),
        "tb_switch_size": abs(d_sw),
        "tb_switch_dir": float(np.sign(d_sw)),
        "tb_balance_size": abs(d_bal),
        "tb_balance_dir": float(np.sign(d_bal)),
    }


with pm.Model() as model:
    sizes = [
        pm.Data("tb_run_size", np.zeros(1, dtype="float64")),
        pm.Data("tb_switch_size", np.zeros(1, dtype="float64")),
        pm.Data("tb_balance_size", np.zeros(1, dtype="float64")),
    ]
    dirs = [
        pm.Data("tb_run_dir", np.zeros(1, dtype="float64")),
        pm.Data("tb_switch_dir", np.zeros(1, dtype="float64")),
        pm.Data("tb_balance_dir", np.zeros(1, dtype="float64")),
    ]
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # How readily a difference in each cue is noticed (per unit of difference).
    log_lam = pm.Normal("log_lam", mu=0.0, sigma=0.7, shape=3)
    lam = pt.exp(log_lam)

    # Shared commitment to the deciding cue, scaled by each person's positive
    # decisiveness (log-normal, non-centred).
    log_a = pm.Normal("log_a", mu=0.5, sigma=0.5)
    sigma_c = pm.HalfNormal("sigma_c", sigma=0.5)
    z_c = pm.Normal("z_c", mu=0.0, sigma=1.0, shape=400)
    commit = pt.exp(log_a + sigma_c * z_c)[participant_id]

    # Person-specific side habit (non-centred).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.3)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = (mu_side + sigma_side * z_side)[participant_id]

    remaining = pt.ones_like(sizes[0])
    p_mix = pt.zeros_like(sizes[0])
    for k in range(3):
        d_k = 1.0 - pt.exp(-lam[k] * sizes[k])
        p_mix = p_mix + remaining * d_k * pm.math.sigmoid(commit * dirs[k] + side)
        remaining = remaining * (1.0 - d_k)
    p_mix = p_mix + remaining * pm.math.sigmoid(side)

    p_left = pm.Deterministic("p_left", pt.clip(p_mix, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
