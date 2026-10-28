# name: interior_density_set_size_gap
# description: For sets with span >=20 whose examples are of mixed parity, yes-rate on interior non-example targets for 3-4 example sets minus 2-example sets; observed above null means sparse (2-example) sets fill their interior less than the model predicts (people need dense examples to generalize across a gap), below null the reverse.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    Xi = np.nan_to_num(X, nan=-1)
    valid = ~np.isnan(X)
    ev = ((np.mod(Xi, 2) == 0) & valid).any(axis=1)
    od = ((np.mod(Xi, 2) == 1) & valid).any(axis=1)
    inner = (t > mn) & (t < mx) & ~is_ex & ((mx - mn) >= 20) & ev & od
    a = inner & (n >= 3)
    b = inner & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
