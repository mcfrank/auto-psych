# name: recency_last_vs_first_example
# description: In 2+-example sets, for non-example targets strictly closer in size to the last-shown example than the first (or vice versa), yes-rate for closer-to-last minus closer-to-first; observed above the null means people weight the most recent example more than the model's fitted recency (below means primacy / less recency than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    first = {s: int(s.split(",")[0]) for s in sets}
    last = {s: int(s.split(",")[-1]) for s in sets}
    k = {s: len(s.split(",")) for s in sets}
    exs = {s: set(int(v) for v in s.split(",")) for s in sets}
    f = ns.map(first).values; la = ns.map(last).values; kk = ns.map(k).values
    t = df["target"].astype(int).values
    ident = np.fromiter((tt in exs[s] for tt, s in zip(t, ns.values)), bool, len(t))
    base = (kk >= 2) & ~ident
    dl, dfst = np.abs(t - la), np.abs(t - f)
    cl, cf = base & (dl < dfst), base & (dfst < dl)
    y = df["rating"].astype(float).values
    if cl.sum() == 0 or cf.sum() == 0:
        return 0.0
    return float(y[cl].mean() - y[cf].mean())
