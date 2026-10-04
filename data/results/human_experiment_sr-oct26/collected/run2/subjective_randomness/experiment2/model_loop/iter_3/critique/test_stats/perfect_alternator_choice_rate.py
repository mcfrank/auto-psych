# name: perfect_alternator_choice_rate
# description: Among pairs where exactly one sequence alternates on every flip (e.g. HTHTHTHT), the rate of choosing the perfect alternator; observed below null means people reject too-regular alternation that the model linearly rewards (model over-produces choosing perfect alternation), above null the reverse.
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
    fa, fb = _feat(df, lambda s: float(_sw(s) == len(s) - 1))
    return _rate_choose_higher(df, fa, fb, np.ones(len(df), bool))
