# name: target_magnitude_slope
# description: OLS slope of the yes response on target/100 among non-example targets more than 10 from every example; positive observed above null means people favor large far targets more than the model (or the model over-produces yes to small numbers), below null means the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = ~is_ex & (dist > 10)
    if m.sum() < 3:
        return 0.0
    x = t[m] / 100.0; yy = y[m]
    vx = np.var(x)
    if vx == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (yy - yy.mean())) / vx)
