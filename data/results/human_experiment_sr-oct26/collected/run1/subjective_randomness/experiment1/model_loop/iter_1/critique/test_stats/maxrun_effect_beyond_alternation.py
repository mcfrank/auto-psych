# name: maxrun_effect_beyond_alternation
# description: Among pairs where both sequences have alternation rate >= 0.5, the coefficient on (longest run of right minus longest run of left) in a linear-probability regression of chose_left on [1, alt_left - alt_right, maxrun_right - maxrun_left]; observed above the null means people penalise long runs beyond what the alternation rate captures, which the model under-predicts.

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
    ra = _feat(df["sequence_a"], _maxrun); rb = _feat(df["sequence_b"], _maxrun)
    y = df["chose_left"].to_numpy(dtype=float)
    m = (aa >= 0.5) & (ab >= 0.5)
    if m.sum() < 5:
        return 0.0
    X = np.column_stack([np.ones(m.sum()), aa[m] - ab[m], rb[m] - ra[m]])
    coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return float(coef[2])
