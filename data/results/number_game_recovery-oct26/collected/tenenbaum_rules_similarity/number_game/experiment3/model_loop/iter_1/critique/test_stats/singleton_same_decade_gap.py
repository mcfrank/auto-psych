# name: singleton_same_decade_gap
# description: For single-example sets, among non-example targets within 9 of the example, yes-rate for targets in the same decade (same (x-1)//10) minus a different decade; observed above null means people use decade boundaries more strongly than the model's decade bonus, below null means the model over-weights decade-aligned ranges.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    last = X[np.arange(len(X)), n - 1]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    x = X[:, 0]
    near = (n == 1) & (np.abs(t - x) <= 9) & (t != x)
    sd = (t - 1) // 10 == (x - 1) // 10
    a = near & sd
    b = near & ~sd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
