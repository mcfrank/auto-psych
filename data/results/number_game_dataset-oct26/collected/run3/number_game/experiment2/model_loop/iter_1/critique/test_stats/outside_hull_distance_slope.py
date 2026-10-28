# name: outside_hull_distance_slope
# description: For sets of 2+ examples and targets outside the examples' min-max range, the OLS slope of yes on distance (in units of 10) to the nearest example; a more negative observed value than the model means people's yes-rate falls off with distance more steeply than predicted (similarity gradient), a less negative/positive one means flatter generalization than predicted.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = (~np.isnan(ex)).sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    m = (n >= 2) & ((t < lo) | (t > hi))
    if m.sum() < 3:
        return 0.0
    d = np.where(t < lo, lo - t, t - hi)[m] / 10.0
    y = df["rating"].to_numpy(dtype=float)[m]
    dc = d - d.mean()
    v = (dc ** 2).sum()
    if v == 0:
        return 0.0
    return float((dc * (y - y.mean())).sum() / v)
