# name: recency_nearest_example
# description: For 2+ example sets and non-member targets, yes-rate when the nearest example is the last one shown minus when it is the first one shown; nonzero observed vs a ~0 null means presentation order (recency/primacy) matters, which the order-invariant model cannot produce.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    d = np.abs(ex - t[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    idx = np.argmin(d, axis=1)
    mind = d[np.arange(len(t)), idx]
    r = df["rating"].astype(float).to_numpy()
    ok = (n >= 2) & (mind > 0)
    last = ok & (idx == n - 1)
    first = ok & (idx == 0)
    if last.sum() == 0 or first.sum() == 0:
        return 0.0
    return float(r[last].mean() - r[first].mean())
