# name: more_switches_among_high_switch_pairs
# description: Among pairs where both sequences switch on at least 60% of transitions and switch rates differ, the rate of choosing the more-switching sequence; observed above null means the ideal-rate penalty over-penalises high switching (model under-produces choosing the more-switching one), below null the penalty is too weak.
import numpy as np

def _feat(df, fn):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    u = {s: fn(s) for s in set(a) | set(b)}
    return np.array([u[s] for s in a], dtype=float), np.array([u[s] for s in b], dtype=float)

def _sw(s):
    return float(sum(1 for x, y in zip(s, s[1:]) if x != y))

def _swr(s):
    return _sw(s) / (len(s) - 1)

def _run(s):
    m = c = 1
    for x, y in zip(s, s[1:]):
        c = c + 1 if x == y else 1
        m = max(m, c)
    return float(m)

def _trip(s):
    return float(len({s[i:i+3] for i in range(len(s) - 2)}))

def _span(s):
    t = hi = lo = 0
    for c in s:
        t += 1 if c == "H" else -1
        hi = max(hi, t); lo = min(lo, t)
    return float(hi - lo) / len(s)

def _rate_choose_higher(df, fa, fb, mask):
    y = df["chose_left"].to_numpy()
    m = mask & (fa != fb)
    if m.sum() == 0:
        return 0.5
    chose_higher = np.where(fa[m] > fb[m], y[m], 1 - y[m])
    return float(chose_higher.mean())


def test_statistic(df):
    fa, fb = _feat(df, _swr)
    return _rate_choose_higher(df, fa, fb, (fa >= 0.6) & (fb >= 0.6))
