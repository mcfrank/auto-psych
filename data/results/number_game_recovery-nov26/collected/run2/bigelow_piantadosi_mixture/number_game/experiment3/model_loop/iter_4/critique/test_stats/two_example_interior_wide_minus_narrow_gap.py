# name: two_example_interior_wide_minus_narrow_gap
# description: For two-example sets, yes rate on non-example targets strictly between the examples when the gap is >=40 minus when the gap is <=20; observed above null means people fill wide gaps more than the model's size-principle ranges predict (observed below null: less).
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
    gap = mx - mn
    base = (n == 2) & ~isin & (t > mn) & (t < mx)
    return m(base & (gap >= 40)) - m(base & (gap <= 20))
