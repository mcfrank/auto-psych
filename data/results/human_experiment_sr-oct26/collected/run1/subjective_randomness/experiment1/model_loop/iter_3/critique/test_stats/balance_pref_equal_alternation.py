# name: balance_pref_equal_alternation
# description: Among trials whose two sequences have identical alternation rates but different H/T imbalance, the proportion choosing the more balanced sequence; the incumbent (alternation only) predicts ~0.5, so observed above null means people use H/T balance, which the model lacks.
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
    ia, ib = _imb(df["sequence_a"]), _imb(df["sequence_b"])
    m = (np.abs(aa - ab) < 1e-9) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    chose_bal = np.where(ia[m] < ib[m], y, 1 - y)
    return float(chose_bal.mean())
