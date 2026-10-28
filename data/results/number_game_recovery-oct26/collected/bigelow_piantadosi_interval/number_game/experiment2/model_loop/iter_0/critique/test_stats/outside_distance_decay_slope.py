# name: outside_distance_decay_slope
# description: For targets outside the examples' [min, max] range, OLS slope of rating on log(1 + distance to the nearest example); more negative observed than null means yes-responses fall off with distance from the examples more steeply than the model predicts (less negative = flatter, broader generalization).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    tg = df["target"].astype(int).values.astype(float)
    mn, mx = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    out = (tg < mn) | (tg > mx)
    dist = np.nanmin(np.abs(ex - tg[:, None]), axis=1)
    x = np.log1p(dist[out])
    y = df["rating"].values.astype(float)[out]
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / x.var())
