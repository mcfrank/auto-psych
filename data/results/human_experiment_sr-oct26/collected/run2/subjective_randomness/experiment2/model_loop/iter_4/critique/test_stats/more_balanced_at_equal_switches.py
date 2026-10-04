# name: more_balanced_at_equal_switches
# description: Among pairs with equal switch counts but different |#H - #T| imbalance, the rate of choosing the more balanced sequence; observed above null means people reward an even heads/tails count more than the tally-span cue implies (model under-produces it), below null the reverse.
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
    sa, sb = _feat(df, _sw)
    ba, bb = _feat(df, lambda s: -abs(s.count("H") - s.count("T")))
    return _rate_choose_higher(df, ba, bb, sa == sb)
