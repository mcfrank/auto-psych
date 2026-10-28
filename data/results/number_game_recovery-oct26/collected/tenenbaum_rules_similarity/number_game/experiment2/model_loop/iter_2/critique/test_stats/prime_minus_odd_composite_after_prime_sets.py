# name: prime_minus_odd_composite_after_prime_sets
# description: On sets of >=2 examples that are all primes, yes rate to non-example prime targets minus yes rate to odd composite targets; observed above null means people pick out the prime rule more sharply than the model, below null means the model over-weights primality relative to oddness/similarity.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    primes = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97], dtype=float)
    exp_ = np.where(np.isnan(ex), 2.0, ex)
    allp = np.all(np.isin(exp_, primes), axis=1) & (n >= 2)
    tp = np.isin(t, primes)
    oc = (t % 2 == 1) & ~tp & (t > 1)
    a = allp & ~is_ex & tp
    b = allp & ~is_ex & oc
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
