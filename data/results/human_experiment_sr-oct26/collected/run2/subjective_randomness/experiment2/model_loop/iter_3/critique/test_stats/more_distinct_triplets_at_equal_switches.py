# name: more_distinct_triplets_at_equal_switches
# description: Among pairs with equal switch counts but different numbers of distinct length-3 substrings, the rate of choosing the sequence with more distinct triplets; observed above null means people reward local pattern variety that span and switching miss (model under-produces it).
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
    sa, sb = _feat(df, _sw)
    ta, tb = _feat(df, lambda s: float(len({s[i:i+3] for i in range(len(s) - 2)})))
    return _rate_choose_higher(df, ta, tb, sa == sb)
