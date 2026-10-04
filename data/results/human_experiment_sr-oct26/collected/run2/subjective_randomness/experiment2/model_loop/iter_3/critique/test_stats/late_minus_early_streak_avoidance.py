# name: late_minus_early_streak_avoidance
# description: Rate of choosing the sequence with the shorter longest run in the second half of each session minus the first half (trial_index split at the participant median); observed away from null (~0) means streak avoidance changes over the session, which the static model cannot produce (below null: it weakens late).
import numpy as np

def _feat(df, fn):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    u = {s: fn(s) for s in set(a) | set(b)}
    return np.array([u[s] for s in a], dtype=float), np.array([u[s] for s in b], dtype=float)

def _sw(s):
    return float(sum(1 for x, y in zip(s, s[1:]) if x != y))

def _run(s):
    m = c = 1
    for x, y in zip(s, s[1:]):
        c = c + 1 if x == y else 1
        m = max(m, c)
    return float(m)

def _rate_choose_higher(df, fa, fb, mask):
    y = df["chose_left"].to_numpy()
    m = mask & (fa != fb)
    if m.sum() == 0:
        return 0.5
    chose_higher = np.where(fa[m] > fb[m], y[m], 1 - y[m])
    return float(chose_higher.mean())

def test_statistic(df):
    ra, rb = _feat(df, _run)
    t = df["trial_index"].to_numpy(dtype=float)
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy(dtype=float)
    return _rate_choose_higher(df, -ra, -rb, t > med) - _rate_choose_higher(df, -ra, -rb, t <= med)
