# name: rule_matching_target_yes
# description: For sets of 3-4 examples that all share a last digit, the yes rate on non-example targets with that same last digit (inside or outside the span); observed above null means people accept rule-consistent targets more fully than the model (it under-commits to the shared-final-digit rule).
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
    Xi = np.nan_to_num(X, nan=-1).astype(int)
    valid = np.isfinite(X)
    ld = Xi % 10
    same_ld = np.all((ld == ld[:, :1]) | ~valid, axis=1)
    mask = (n >= 3) & same_ld & ~isin & (t % 10 == ld[:, 0])
    return m(mask)
