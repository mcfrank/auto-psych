# name: singleton_small_vs_large_near_gap
# description: For single-example sets, yes-rate on targets 1-10 away from an example <=20 minus that for an example >=50; observed below null means people generalize less (in absolute distance) around small numbers than the model's linear-range account predicts (Weber-like log-magnitude similarity), above null the reverse.
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
    near = (n == 1) & (np.abs(t - x) >= 1) & (np.abs(t - x) <= 10)
    a = near & (x <= 20)
    b = near & (x >= 50)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
