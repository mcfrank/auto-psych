# name: late_minus_early_inside_hull
# description: Yes-rate for non-example targets inside the example range on late trials (trial_index>=300) minus early trials (trial_index<150), multi-example sets only; the model attributes session drift to blind-yes mainly on excluded targets, so observed far from model means in-range acceptance changes over the session differently (observed < model: in-range yes falls or stays flat late).
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
    tr = df["trial_index"].astype(float).to_numpy()
    m = (n >= 2) & ~member & (t > lo) & (t < hi)
    return rate(m & (tr >= 300)) - rate(m & (tr < 150))
