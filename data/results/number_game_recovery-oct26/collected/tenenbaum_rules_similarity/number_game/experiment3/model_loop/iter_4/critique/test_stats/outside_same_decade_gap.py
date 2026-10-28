# name: outside_same_decade_gap
# description: For sets of >=2 examples, among non-example targets 1-9 outside the span, yes-rate when the target is in the same decade (1-10, 11-20, ...) as the nearest edge example minus when it crosses into another decade; observed above null means decade boundaries limit generalization more than the model's interval decade bonus captures, below null less.
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
    edge = np.where(t < mn, mn, mx)
    same = ((t - 1) // 10) == ((edge.astype(int) - 1) // 10)
    near = (n >= 2) & (dist >= 1) & (dist <= 9) & ~is_ex
    a = near & same
    b = near & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
