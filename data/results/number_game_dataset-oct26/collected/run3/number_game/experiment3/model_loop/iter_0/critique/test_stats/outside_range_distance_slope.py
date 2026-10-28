# name: outside_range_distance_slope
# description: Among targets outside the [min,max] range of their set, the OLS slope of yes against log(1 + distance to the nearest example); observed more negative than null means human generalization falls off with distance more steeply than the model predicts (less negative = flatter).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ex = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    E = ex[codes]  # (rows, 4) examples, NaN-padded, in shown order
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    out = (t < lo) | (t > hi)
    d = np.log1p(np.nanmin(np.abs(E - t[:, None]), axis=1))
    x, yy = d[out], y[out]
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, yy, bias=True)[0, 1] / np.var(x))
