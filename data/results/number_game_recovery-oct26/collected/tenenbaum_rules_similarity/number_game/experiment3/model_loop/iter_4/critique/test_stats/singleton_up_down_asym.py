# name: singleton_up_down_asym
# description: For single-example sets with the example in 11-90, yes-rate on targets 1-10 above the example minus targets 1-10 below it; observed above null means people extend a lone example upward more than downward beyond what the model (symmetric ranges) predicts, below null the reverse.
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
    x = mx
    base = (n == 1) & (x >= 11) & (x <= 90)
    a = base & (t - x >= 1) & (t - x <= 10)
    b = base & (x - t >= 1) & (x - t <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
