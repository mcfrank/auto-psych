# name: shared_last_digit_boost_outside_hull
# description: Among targets outside the example range (excluding examples), yes-rate for targets sharing a last digit with some example minus yes-rate for those that do not; observed > model means surface digit similarity drives generalization more than the model's ends-in-d rules capture.
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
    nsa = ns.to_numpy()
    shares = np.array([(tt % 10) in {x % 10 for x in parsed[s]} for s, tt in zip(nsa, t)])
    out = ((t < lo) | (t > hi)) & ~member
    return rate(out & shares) - rate(out & ~shares)
