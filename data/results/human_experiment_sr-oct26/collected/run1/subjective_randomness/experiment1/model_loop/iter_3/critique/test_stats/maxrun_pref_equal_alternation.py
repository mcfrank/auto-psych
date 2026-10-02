# name: maxrun_pref_equal_alternation
# description: Among trials with alternation rates within 0.15 of each other but different longest runs, the proportion choosing the sequence with the shorter longest run; observed above null means people penalise long streaks beyond what alternation rate captures (model under-produces streak aversion).
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
    ra, rb = _maxrun(df["sequence_a"]), _maxrun(df["sequence_b"])
    m = (np.abs(aa - ab) <= 0.15) & (ra != rb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.where(ra[m] < rb[m], y, 1 - y).mean())
