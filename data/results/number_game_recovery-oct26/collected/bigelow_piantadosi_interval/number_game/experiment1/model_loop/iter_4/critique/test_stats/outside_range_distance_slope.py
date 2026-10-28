# name: outside_range_distance_slope
# description: For sets of 2+ examples, OLS slope of yes on log(1 + distance to the nearest range endpoint) across non-example targets outside the range; more negative observed than null means people's generalization falls off with distance more steeply than the model's (model too flat), less negative means the model falls off too steeply.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"])
    E = np.full((len(uniq), 4), -1000)
    for i, s in enumerate(uniq):
        v = [int(x) for x in str(s).split(",")]
        E[i, : len(v)] = v
    arr = E[codes]
    valid = arr > 0
    n_ex = valid.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    is_ex = (arr == t[:, None]).any(axis=1)
    lo = np.where(valid, arr, 1000).min(axis=1)
    hi = np.where(valid, arr, -1000).max(axis=1)
    dist = np.abs(np.where(valid, arr, 10000) - t[:, None]).min(axis=1)
    out = ((t < lo) | (t > hi)) & ~is_ex & (n_ex >= 2)
    x = np.log1p(dist[out].astype(float))
    y = r[out]
    if len(x) < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
