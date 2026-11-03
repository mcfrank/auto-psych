# name: single_example_above_minus_below
# description: For one-example sets with the example in 15-85, yes rate on targets 1-10 above the example minus targets 1-10 below it; observed above null means people extend upward more than the model's symmetric equal-range prior predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    for c in range(x.shape[1], 4):
        x[c] = np.nan
    x = x.iloc[:, :4]
    X = x.to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.isfinite(X).sum(axis=1)
    mn = np.nanmin(X, axis=1)
    mx = np.nanmax(X, axis=1)
    isin = (X == t[:, None]).any(axis=1)
    nd = np.nanmin(np.abs(X - t[:, None]), axis=1)
    first = X[:, 0]
    last = X[np.arange(len(X)), np.maximum(n - 1, 0)]
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    d = t - first
    base = (n == 1) & (first >= 15) & (first <= 85)
    return m(base & (d >= 1) & (d <= 10)) - m(base & (d <= -1) & (d >= -10))
