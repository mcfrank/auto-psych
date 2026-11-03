# name: recency_minus_primacy_extrapolation
# description: For sets of 2+ examples and targets outside the examples' [min,max] span, yes rate for targets strictly closer to the last-shown example minus yes rate for targets strictly closer to the first-shown one; observed above null means people generalize more around the most recent example than the order-blind model predicts (recency).
def _m(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.sum(~np.isnan(X), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    first = X[:, 0]
    last = X[np.arange(len(X)), n - 1]
    out = (n >= 2) & ((t < mn) | (t > mx))
    dl = np.abs(t - last)
    dfst = np.abs(t - first)
    return _m(y, out & (dl < dfst)) - _m(y, out & (dfst < dl))
