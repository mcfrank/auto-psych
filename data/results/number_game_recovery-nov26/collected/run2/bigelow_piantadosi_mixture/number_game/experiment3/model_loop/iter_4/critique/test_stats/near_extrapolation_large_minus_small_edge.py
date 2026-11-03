# name: near_extrapolation_large_minus_small_edge
# description: For targets outside the examples' span at distance 1-10 from the nearest example, yes rate when that nearest example is >=50 minus when it is <=25; observed above null means people generalize further around large numbers (Weber-like relative width) than the model's equal-range prior predicts.
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
    out = (t < mn) | (t > mx)
    near = np.where(t < mn, mn, mx)
    base = out & (nd >= 1) & (nd <= 10)
    return m(base & (near >= 50)) - m(base & (near <= 25))
