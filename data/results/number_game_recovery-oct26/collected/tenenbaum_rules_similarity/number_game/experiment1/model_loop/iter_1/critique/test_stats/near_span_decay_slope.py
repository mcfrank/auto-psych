# name: near_span_decay_slope
# description: Among targets outside the examples range and within 40 of it, OLS slope of yes on distance-to-range (per 10 units); observed less negative than null_mean means people generalize to nearby numbers more gradually (flatter) than the model, more negative means a sharper fall-off.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    dist = np.where(t > mx, t - mx, np.where(t < mn, mn - t, 0.0))
    m = (dist > 0) & (dist <= 40)
    if m.sum() < 3:
        return 0.0
    x = dist[m] / 10.0; y = r[m]
    x = x - x.mean(); v = (x * x).sum()
    return float((x * (y - y.mean())).sum() / v) if v > 0 else 0.0
