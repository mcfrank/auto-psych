# name: inside_range_by_setsize_slope
# description: Slope, across number of examples (2-4), of the yes rate for non-example targets strictly inside the examples' min-max range; observed above null_mean means people fill in the spanned range increasingly with more examples than the model predicts, below means the model's size-principle tightening is too weak/strong in the opposite way.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n_ex = np.sum(~np.isnan(E), axis=1)
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    notex = ~np.any(E == t[:, None], axis=1)
    m = (n_ex >= 2) & notex & (t > lo) & (t < hi)
    xs, ys = [], []
    for k in (2, 3, 4):
        s = m & (n_ex == k)
        if s.any():
            xs.append(k)
            ys.append(r[s].mean())
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, float), np.array(ys), 1)[0])
