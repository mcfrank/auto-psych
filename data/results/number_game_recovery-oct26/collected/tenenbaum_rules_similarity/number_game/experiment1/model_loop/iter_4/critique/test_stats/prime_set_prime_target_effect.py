# name: prime_set_prime_target_effect
# description: For sets of 2+ examples that are all odd primes, among non-example odd targets more than 5 from every example, yes rate for primes minus non-primes; observed above null_mean means people use the primes rule more than the model (which pools it with even/odd/squares/cubes) predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    isp = np.isin(np.where(valid, ex, -1), primes) & (np.where(valid, ex, 1) != 2)
    allp = np.all(np.where(valid, isp, True), axis=1) & (n_ex >= 2)
    base = allp & (~member) & (dist > 5) & (np.mod(t, 2) == 1)
    tp = np.isin(t, primes)
    a = base & tp; b = base & (~tp)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
