# name: short_vs_long_alt_pref
# description: Proportion choosing the higher-alternation sequence among length<=5 trials minus the same among length-8 trials (pairs with different alternation rates); observed above null means short sequences are judged by alternation more than the length-independent model predicts, below means less.
def _alt(s):
    u = pd.unique(s)
    m = {x: sum(a != b for a, b in zip(x, x[1:])) / (len(x) - 1) for x in u}
    return s.map(m).to_numpy(dtype=float)


def _imb(s):
    return np.abs(s.str.count("H").to_numpy() / s.str.len().to_numpy() - 0.5)


def _maxrun(s):
    u = pd.unique(s)
    def mr(x):
        best = cur = 1
        for a, b in zip(x, x[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        return best
    return s.map({x: mr(x) for x in u}).to_numpy(dtype=float)



def test_statistic(df):
    aa, ab = _alt(df["sequence_a"]), _alt(df["sequence_b"])
    L = df["sequence_a"].str.len().to_numpy()
    y = df["chose_left"].to_numpy()
    d = aa != ab
    ch = np.where(aa > ab, y, 1 - y)
    s = d & (L <= 5)
    l = d & (L == 8)
    if s.sum() == 0 or l.sum() == 0:
        return 0.0
    return float(ch[s].mean() - ch[l].mean())
