# name: first_minus_last_example_proximity
# description: On sets of >=2 examples, yes rate to non-example targets within 5 of the first-shown example (and nearer it than the last) minus that for targets within 5 of the last-shown example (and nearer it); the order-blind model predicts ~0 beyond stimulus effects, so observed above null means primacy (first example weighted more), below null means recency.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    last = ex[np.arange(len(ex)), np.maximum(n - 1, 0)]
    first = ex[:, 0]
    d1 = np.abs(t - first); dl = np.abs(t - last)
    ok = (n >= 2) & ~is_ex
    a = ok & (d1 <= 5) & (d1 < dl)
    b = ok & (dl <= 5) & (dl < d1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
