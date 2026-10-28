# name: outside_span_near_minus_far
# description: Among targets outside the example span, yes-rate for targets 1-10 beyond the nearest span edge minus targets 25-60 beyond it; observed above the null means people's generalization falls off with distance outside the span more steeply than the model's single magnitude tuning predicts (below means a flatter falloff than modelled).
def test_statistic(df):
    ns = df["number_set"].astype(str)
    sets = ns.unique()
    mx = {s: max(int(v) for v in s.split(",")) for s in sets}
    mn = {s: min(int(v) for v in s.split(",")) for s in sets}
    hi, lo = ns.map(mx).values, ns.map(mn).values
    t = df["target"].astype(int).values
    d = np.maximum(0, np.maximum(lo - t, t - hi))
    y = df["rating"].astype(float).values
    a = (d >= 1) & (d <= 10)
    b = (d >= 25) & (d <= 60)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
