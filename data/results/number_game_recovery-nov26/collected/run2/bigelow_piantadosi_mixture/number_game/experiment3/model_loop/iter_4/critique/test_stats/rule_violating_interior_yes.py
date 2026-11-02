# name: rule_violating_interior_yes
# description: For sets of 3-4 examples that all share a last digit or are all multiples of some k in 3-12, the yes rate on non-example targets inside the examples' span that break every such shared rule; observed below null means people reject in-span rule violators more strongly than the model (its range hypotheses over-generalize when a rule fits).
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
    ok = np.zeros(len(X), bool)
    viol = np.ones(len(X), bool)
    ld = Xi % 10
    same_ld = np.all((ld == ld[:, :1]) | ~valid, axis=1)
    ok |= same_ld
    viol &= ~(same_ld & (t % 10 == ld[:, 0]))
    for k in range(3, 13):
        allk = np.all((Xi % k == 0) | ~valid, axis=1)
        ok |= allk
        viol &= ~(allk & (t % k == 0))
    mask = (n >= 3) & ok & viol & ~isin & (t > mn) & (t < mx)
    return m(mask)
