# name: recency_out_of_range_yes
# description: Yes rate for targets outside the examples' [min,max] span (sets of 2+ examples) that lie strictly closer to the last-shown example than to the first-shown one; observed above null means people generalize more around the most recent example than the order-blind model predicts (recency).
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
    m = (n >= 2) & ((t < mn) | (t > mx)) & (np.abs(t - last) < np.abs(t - first))
    return float(y[m].mean()) if m.sum() > 0 else 0.0
