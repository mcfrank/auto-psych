# name: intermediate_cell_fraction
# description: Fraction of (set,target) cells with >=4 responses whose observed yes-proportion lies strictly between 0.25 and 0.75; observed > model means responses are more graded/inconsistent across trials than the model's near-deterministic concept posterior predicts, observed < model means more all-or-none.
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
    g = pd.DataFrame({"c": ns.to_numpy() + "|" + pd.Series(t).astype(str).to_numpy(), "y": y})
    agg = g.groupby("c")["y"].agg(["mean", "size"])
    agg = agg[agg["size"] >= 4]
    if len(agg) == 0:
        return 0.0
    return float(((agg["mean"] > 0.25) & (agg["mean"] < 0.75)).mean())
