# name: upward_vs_downward_extension
# description: For sets of 2+ examples, yes-rate on targets 1-20 above the largest example minus yes-rate on targets 1-20 below the smallest example; observed > model means people extend the range upward more readily than the model predicts (asymmetric magnitude generalization), observed < model means they extend downward more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    up = (n >= 2) & (t > hi) & (t - hi <= 20)
    dn = (n >= 2) & (t < lo) & (lo - t <= 20)
    if not up.any() or not dn.any():
        return 0.0
    return float(y[up].mean() - y[dn].mean())
