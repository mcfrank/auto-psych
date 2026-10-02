# name: balance_pref_matched_alternation
# description: Among pairs of irregular sequences (both alternation rates >= 0.5) whose alternation rates differ by at most 1/7 but whose H/T imbalance differs, the proportion choosing the more balanced sequence; observed above the null means the model under-predicts a preference for balanced H/T counts that alternation alone cannot capture.

def _alt(s):
    return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)

def _imb(s):
    return abs(s.count("H") - len(s) / 2.0) / len(s)

def _maxrun(s):
    best = cur = 1
    for x, y in zip(s, s[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best

def _feat(col, f):
    u = pd.unique(col)
    return col.map({s: f(s) for s in u}).to_numpy(dtype=float)

def test_statistic(df):
    aa = _feat(df["sequence_a"], _alt); ab = _feat(df["sequence_b"], _alt)
    ia = _feat(df["sequence_a"], _imb); ib = _feat(df["sequence_b"], _imb)
    y = df["chose_left"].to_numpy(dtype=float)
    m = (aa >= 0.5) & (ab >= 0.5) & (np.abs(aa - ab) <= 1/7 + 1e-9) & (np.abs(ia - ib) > 1e-9)
    if m.sum() == 0:
        return 0.5
    left_more_balanced = ia[m] < ib[m]
    chose_bal = np.where(left_more_balanced, y[m], 1 - y[m])
    return float(chose_bal.mean())
