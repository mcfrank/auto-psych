# name: set_size_near_outside_gap
# description: Yes-rate for non-example targets 1-10 outside the span in sets of 3-4 examples minus the same in sets of 2; observed below null means more examples tighten generalization more than the model's size principle predicts, above null means people tighten less with more examples.
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
    near = (dist >= 1) & (dist <= 10)
    a = near & (n >= 3)
    b = near & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
