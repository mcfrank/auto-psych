# name: last_vs_first_example_proximity
# description: Among non-member targets of multi-example sets whose first and last shown examples differ by >=10, yes-rate for targets strictly closer to the last-shown example minus yes-rate for targets strictly closer to the first-shown example; observed > model means a recency/order effect the order-blind model under-produces (observed < model means primacy).
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
    m = (n >= 2) & ~member & (np.abs(last - first) >= 10)
    dl = np.abs(t - last); dfst = np.abs(t - first)
    return rate(m & (dl < dfst)) - rate(m & (dfst < dl))
