# name: participant_spread_alt_pref
# description: Across participants, the standard deviation of each person's rate of choosing the higher-alternation sequence on pairs whose alternation rates differ by >= 0.25; observed below the null means the model over-produces individual differences in ideal alternation rate (people are more homogeneous than it assumes), above means under-produces.

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
    y = df["chose_left"].to_numpy(dtype=float)
    m = np.abs(aa - ab) >= 0.25
    ch = np.where(aa > ab, y, 1 - y)
    s = pd.Series(ch[m]).groupby(df["participant_id"].to_numpy()[m]).mean()
    if len(s) < 2:
        return 0.0
    return float(s.std(ddof=1))
