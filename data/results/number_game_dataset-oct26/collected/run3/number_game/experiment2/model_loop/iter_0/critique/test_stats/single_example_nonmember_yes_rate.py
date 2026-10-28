# name: single_example_nonmember_yes_rate
# description: Yes-rate for non-example targets on single-example sets; observed > model means people generalize more broadly from one example than the model's size-principle posterior (with n**gamma evidence weighting) predicts, observed < model means more conservative.
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
    return rate((n == 1) & ~member)
