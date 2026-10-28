# name: outside_gradient_steepness
# description: In sets of >=2, yes-rate for targets 1-3 outside the example span minus yes-rate for targets 7-12 outside the span (either side); observed above null means people's generalization falls off more steeply just beyond the edges than the model predicts, below null means a shallower gradient than the model.
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
    dist = np.maximum(mn - t, t - mx)
    g = n >= 2
    a = g & (dist >= 1) & (dist <= 3)
    b = g & (dist >= 7) & (dist <= 12)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
