# name: first_vs_last_example_proximity
# description: Among non-example targets within 10 of some example in multi-example sets, yes rate when the nearest example is the first shown minus when it is the last shown; the order-invariant model predicts ~0, so observed above null means a primacy effect (first example weighted more), below null a recency effect.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    n = np.sum(~np.isnan(ex), axis=1)
    d = np.where(np.isnan(dist), np.inf, dist)
    idx = np.argmin(d, axis=1)
    m = nonmember & (n >= 2) & (nearest <= 10)
    a, b = r[m & (idx == 0)], r[m & (idx == n - 1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
