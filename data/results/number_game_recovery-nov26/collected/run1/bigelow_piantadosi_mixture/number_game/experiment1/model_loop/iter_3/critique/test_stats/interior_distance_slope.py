# name: interior_distance_slope
# description: Among targets inside the range of 3-4-example sets spanning >= 40 (non-examples), the least-squares slope of yes against distance to the nearest example (per unit distance); observed more negative than null means people's in-range generalisation decays with distance from examples more than the model (whose intervals make the interior flat), less negative means flatter.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.isfinite(ex).sum(axis=1)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (n >= 3) & (hi - lo >= 40) & (t > lo) & (t < hi) & (d > 0)
    if m.sum() < 3:
        return 0.0
    x = d[m]; y = r[m]
    vx = ((x - x.mean()) ** 2).sum()
    if vx == 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)
