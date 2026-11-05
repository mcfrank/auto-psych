# name: last_minus_first_nearest_example
# description: For non-example targets on sets of 2+ examples, yes rate when the numerically nearest example is the last-shown one minus when it is the first-shown one; observed above null means a recency effect (the last example pulls generalization more than the order-invariant model predicts), below null a primacy effect.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy().astype(float)
    n_ex = np.sum(~np.isnan(ex), axis=1)
    is_ex = (ex == t[:, None]).any(axis=1)
    d = np.abs(ex - t[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    dmin = d.min(axis=1)
    ties = (d == dmin[:, None]).sum(axis=1) > 1
    ok = (n_ex >= 2) & ~is_ex & ~ties
    r = df["rating"].to_numpy().astype(float)
    last = ok & (nearest == n_ex - 1)
    first = ok & (nearest == 0)
    if last.sum() == 0 or first.sum() == 0:
        return 0.0
    return float(r[last].mean() - r[first].mean())
