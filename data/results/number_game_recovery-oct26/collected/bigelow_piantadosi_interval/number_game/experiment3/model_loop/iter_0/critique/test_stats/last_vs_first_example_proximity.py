# name: last_vs_first_example_proximity
# description: In 3-4 example sets, yes rate for non-example targets whose nearest example is the LAST shown minus those whose nearest is the FIRST shown; observed above null means a recency advantage the first-example-anchored model misses, below null means primacy stronger than modeled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.abs(ex - t[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    notex = ~np.any(ex == t[:, None], axis=1)
    sel = (n >= 3) & notex
    a = sel & (nearest == n - 1)
    b = sel & (nearest == 0)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
