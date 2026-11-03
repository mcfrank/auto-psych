# name: interior_yes_4ex_minus_2ex_matched_span
# description: Yes rate on non-example targets inside the span for 4-example sets minus 2-example sets, both restricted to spans of 20-60; observed above null means extra examples tighten in-span generalization less (or raise it more) than the model predicts.
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
    span = mx - mn
    base = ~isin & (t > mn) & (t < mx) & (span >= 20) & (span <= 60)
    return m(base & (n == 4)) - m(base & (n == 2))
