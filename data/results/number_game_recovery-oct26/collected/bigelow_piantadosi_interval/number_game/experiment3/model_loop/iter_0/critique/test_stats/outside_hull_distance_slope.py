# name: outside_hull_distance_slope
# description: For targets outside the examples' [min,max] range, least-squares slope of yes (0/1) on log(1 + distance to nearest range edge); observed more negative than null means human generalization falls off with distance beyond the examples more steeply than the model predicts (less negative = flatter).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    sel = d > 0
    if sel.sum() < 3:
        return 0.0
    x = np.log1p(d[sel]); y = r[sel]
    vx = np.var(x)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx) if vx > 0 else 0.0
