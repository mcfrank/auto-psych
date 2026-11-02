# name: outside_range_distance_slope
# description: Among targets outside [min,max] of the examples, the OLS slope of yes on log(1 + distance to the nearest example); observed less negative than null means people's similarity gradient decays more slowly than the model's interval generalisation, more negative means a sharper fall-off.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (t < lo) | (t > hi)
    if m.sum() < 3:
        return 0.0
    x = np.log1p(d[m]); y = df["rating"].astype(float).to_numpy()[m]
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])
