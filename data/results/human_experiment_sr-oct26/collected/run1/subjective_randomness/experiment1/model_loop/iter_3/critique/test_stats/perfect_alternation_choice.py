# name: perfect_alternation_choice
# description: Among trials where exactly one sequence is perfectly alternating (alternation rate 1), the proportion choosing it; observed below null means people reject perfect alternation (too regular) more than the quadratic ideal-distance model predicts, above means they favour it more.
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
    y = df["chose_left"].to_numpy()
    m = (aa == 1) ^ (ab == 1)
    if m.sum() == 0:
        return 0.5
    return float(np.where(aa[m] == 1, y[m], 1 - y[m]).mean())
