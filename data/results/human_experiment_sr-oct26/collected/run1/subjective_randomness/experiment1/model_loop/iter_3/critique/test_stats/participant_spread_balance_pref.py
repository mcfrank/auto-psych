# name: participant_spread_balance_pref
# description: SD across participants of each participant's proportion choosing the more balanced sequence among pairs whose imbalance differs by >= 0.2; observed above null means people differ in balance weighting more than the model (no balance term) allows.
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
    ia, ib = _imb(df["sequence_a"]), _imb(df["sequence_b"])
    m = np.abs(ia - ib) >= 0.2
    y = df["chose_left"].to_numpy()[m]
    cb = np.where(ia[m] < ib[m], y, 1 - y)
    g = pd.Series(cb).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(g.std(ddof=0)) if len(g) > 1 else 0.0
