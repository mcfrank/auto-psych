# name: more_heads_choice_rate
# description: Among pairs whose head counts differ, the rate of choosing the sequence with more H; observed away from the null mean means an H-versus-T asymmetry the symmetric span/switch model cannot produce (above: model under-produces a heads preference; below: tails preference).
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
    fa, fb = _feat(df, lambda s: float(s.count("H")))
    return _rate_choose_higher(df, fa, fb, np.ones(len(df), bool))
