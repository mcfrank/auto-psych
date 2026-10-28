# name: above_minus_below_hull_asymmetry
# description: Among targets 1-20 units outside the example range (any set size), yes-rate for targets above the max minus yes-rate for targets below the min; observed > model means people extend upward (larger numbers) more than the symmetric-in-linear-space model predicts, observed < model means downward extension is under-predicted.
def test_statistic(df):
    u = pd.Series(df["number_set"].astype(str).unique())
    parsed = {s: [int(x) for x in s.split(",")] for s in u}
    ns = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = ns.map(lambda s: len(parsed[s])).to_numpy()
    lo = ns.map(lambda s: min(parsed[s])).to_numpy()
    hi = ns.map(lambda s: max(parsed[s])).to_numpy()
    first = ns.map(lambda s: parsed[s][0]).to_numpy()
    last = ns.map(lambda s: parsed[s][-1]).to_numpy()
    member = np.array([tt in parsed[s] for s, tt in zip(ns.to_numpy(), t)])
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    above = (t > hi) & (t - hi <= 20)
    below = (t < lo) & (lo - t <= 20)
    return rate(above) - rate(below)
