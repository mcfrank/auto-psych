# name: proximity_slope
# description: OLS slope of the yes response on distance (1-15) from the target to the nearest example, excluding targets that are examples; more negative observed than null means people's similarity gradient is steeper than the model's, less negative means the model's falloff is too steep.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (d >= 1) & (d <= 15)
    if m.sum() < 3:
        return 0.0
    x = d[m]
    y = df["rating"].to_numpy(dtype=float)[m]
    xc = x - x.mean()
    v = (xc ** 2).sum()
    if v == 0:
        return 0.0
    return float((xc * (y - y.mean())).sum() / v)
