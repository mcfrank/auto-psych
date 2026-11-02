# name: recency_endpoint_proximity
# description: Among targets 1-20 outside the span of a >=2-example set, yes rate when the nearest span endpoint was the last example shown minus when it was the first shown; the model is order-blind, so observed above null means a recency boost in generalization (below means primacy).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), np.maximum(n - 1, 0)]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    below = (t < lo) & (lo - t <= 20)
    above = (t > hi) & (t - hi <= 20)
    endpoint = np.where(below, lo, np.where(above, hi, np.nan))
    ok = (n >= 2) & (below | above)
    is_last = ok & (endpoint == last)
    is_first = ok & (endpoint == first)
    if is_last.sum() == 0 or is_first.sum() == 0:
        return 0.0
    return float(y[is_last].mean() - y[is_first].mean())
