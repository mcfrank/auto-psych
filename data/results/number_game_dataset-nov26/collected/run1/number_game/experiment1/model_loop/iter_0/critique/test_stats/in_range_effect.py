# name: in_range_effect
# description: Among sets with >=2 examples, yes-rate for targets inside [min example, max example] minus yes-rate for targets outside that range; observed above null_mean means people favour interpolation within the example span more than the model (interval-like generalization under-produced), below means the model over-weights range.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).values
    y = df["rating"].astype(float).values
    m = n >= 2
    inside = (t >= lo) & (t <= hi)
    a = m & inside; b = m & ~inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
