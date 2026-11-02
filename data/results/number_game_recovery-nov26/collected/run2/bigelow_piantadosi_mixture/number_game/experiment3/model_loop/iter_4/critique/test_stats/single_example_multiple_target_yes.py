# name: single_example_multiple_target_yes
# description: For one-example sets with example 3-25, yes rate on targets that are multiples of the example (excluding the example) minus yes rate on non-multiple targets more than 10 away; observed above null means people generalize to multiples of a lone example more than the model's rule prior predicts.
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
    e = np.nan_to_num(first, nan=1).astype(int)
    e = np.maximum(e, 1)
    base = (n == 1) & (first >= 3) & (first <= 25) & ~isin
    mult = (t % e == 0)
    return m(base & mult) - m(base & ~mult & (nd > 10))
