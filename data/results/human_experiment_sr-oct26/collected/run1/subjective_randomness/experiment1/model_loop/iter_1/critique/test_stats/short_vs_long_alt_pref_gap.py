# name: short_vs_long_alt_pref_gap
# description: Rate of choosing the higher-alternation sequence (pairs with alternation difference >= 0.25) on short sequences (length <= 7) minus the same rate on length-8 sequences; a discrepancy from the null means the model mis-predicts how alternation preference changes with sequence length (positive observed-minus-null = people prefer alternation more on short sequences than the model predicts).

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
    L = df["sequence_a"].str.len().to_numpy()
    y = df["chose_left"].to_numpy(dtype=float)
    m = np.abs(aa - ab) >= 0.25
    ch = np.where(aa > ab, y, 1 - y)
    s = m & (L <= 7); l = m & (L == 8)
    if s.sum() == 0 or l.sum() == 0:
        return 0.0
    return float(ch[s].mean() - ch[l].mean())
