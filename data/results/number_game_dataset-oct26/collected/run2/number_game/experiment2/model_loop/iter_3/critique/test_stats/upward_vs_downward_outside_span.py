# name: upward_vs_downward_outside_span
# description: Over all sets, yes-rate for targets 1-20 above the largest example minus targets 1-20 below the smallest example; observed above the null means people generalize upward (to larger numbers) more than the model's symmetric outside-span size mismatch predicts (below means a stronger downward bias than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    mx = {s: max(int(v) for v in s.split(",")) for s in sets}
    mn = {s: min(int(v) for v in s.split(",")) for s in sets}
    hi, lo = ns.map(mx).values, ns.map(mn).values
    t = df["target"].astype(int).values
    up = (t > hi) & (t - hi <= 20)
    dn = (t < lo) & (lo - t <= 20)
    y = df["rating"].astype(float).values
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(y[up].mean() - y[dn].mean())
