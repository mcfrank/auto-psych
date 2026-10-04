# name: participant_switch_preference_variance
# description: Across participants, the variance of each person's rate of choosing the sequence whose switch rate is closer to 0.6 (pairs where the distances differ); observed above null means individual differences in switching preference are larger than the model's person-level kappa produces, below null smaller.
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
    fa, fb = _feat(df, lambda s: -abs(_swr(s) - 0.6))
    y = df["chose_left"].to_numpy()
    m = fa != fb
    c = np.where(fa[m] > fb[m], y[m], 1 - y[m])
    r = pd.Series(c).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(r.var())
