# name: switch_preference_long_minus_short
# description: Rate of choosing the more-switching sequence on length-8 pairs minus that rate on pairs of length <=7; observed above null means the alternation preference grows with length more than the model (whose switch weight ignores length) predicts, below null that it shrinks.
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
    fa, fb = _feat(df, lambda s: _sw(s) / (len(s) - 1))
    L = df["sequence_a"].str.len().to_numpy()
    return _rate_choose_higher(df, fa, fb, L == 8) - _rate_choose_higher(df, fa, fb, L <= 7)
