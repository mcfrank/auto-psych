# name: periodic_motif_choice_rate
# description: Among length>=6 pairs where exactly one sequence repeats a period-3 or period-4 motif (e.g. HHTHHT, HTTTHTTT; non-alternating), the rate of choosing the periodic one; observed below null means people detect and reject repeating motifs beyond span and switching (model over-produces choosing them).
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


def _periodic(s):
    if len(s) < 6 or _sw(s) == len(s) - 1 or _sw(s) == 0:
        return 0.0
    for p in (3, 4):
        if all(s[i] == s[i + p] for i in range(len(s) - p)):
            return 1.0
    return 0.0

def test_statistic(df):
    fa, fb = _feat(df, _periodic)
    return _rate_choose_higher(df, fa, fb, np.ones(len(df), dtype=bool))
