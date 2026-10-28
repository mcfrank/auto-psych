# name: recency_inside_span
# description: In 2+-example sets, among non-example targets strictly inside the span (min < t < max), yes-rate for those closer in size to the last-shown example than to the first minus the reverse; observed above the null means recency of the last example boosts generalization inside the span more than the model's memory-retention weighting produces (below means primacy / less recency than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    ex = {s: [int(v) for v in s.split(",")] for s in sets}
    f = ns.map({s: e[0] for s, e in ex.items()}).values
    la = ns.map({s: e[-1] for s, e in ex.items()}).values
    hi = ns.map({s: max(e) for s, e in ex.items()}).values
    lo = ns.map({s: min(e) for s, e in ex.items()}).values
    kk = ns.map({s: len(e) for s, e in ex.items()}).values
    t = df["target"].astype(int).values
    exset = {s: set(e) for s, e in ex.items()}
    ident = np.fromiter((tt in exset[s] for tt, s in zip(t, ns.values)), bool, len(t))
    base = (kk >= 2) & ~ident & (t > lo) & (t < hi)
    dl, dfst = np.abs(t - la), np.abs(t - f)
    cl, cf = base & (dl < dfst), base & (dfst < dl)
    y = df["rating"].astype(float).values
    if cl.sum() == 0 or cf.sum() == 0:
        return 0.0
    return float(y[cl].mean() - y[cf].mean())
