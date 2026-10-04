# name: pair_consensus_extremity
# description: Mean over distinct (unordered) pairs of |rate of choosing the alphabetically-first sequence - 0.5|; observed above null means people agree on specific pairs more strongly than the model predicts (model under-produces stimulus-level consensus), below null the model is too confident per pair.
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
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    y = df["chose_left"].to_numpy()
    first = np.where(a <= b, a, b); second = np.where(a <= b, b, a)
    chose_first = np.where(a <= b, y, 1 - y)
    g = pd.DataFrame({"k1": first, "k2": second, "c": chose_first}).groupby(["k1", "k2"])["c"].mean()
    return float(np.abs(g.to_numpy() - 0.5).mean())
