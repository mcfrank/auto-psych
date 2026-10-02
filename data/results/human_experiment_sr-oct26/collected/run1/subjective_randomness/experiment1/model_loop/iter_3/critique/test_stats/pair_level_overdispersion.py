# name: pair_level_overdispersion
# description: Variance across unordered stimulus pairs of the proportion choosing the lexicographically-first sequence; observed above null means stimulus-level preferences are more extreme/varied than the alternation-distance model produces (model misses some stimulus property), below means less.
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
    a, b = df["sequence_a"].to_numpy(), df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    y = df["chose_left"].to_numpy()
    chose_first = np.where(a < b, y, 1 - y)
    key = pd.Series(first) + "|" + pd.Series(second)
    g = pd.Series(chose_first).groupby(key.to_numpy()).mean()
    return float(g.var(ddof=0))
