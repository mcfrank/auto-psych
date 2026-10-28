# name: prime_set_prime_target_contrast
# description: For sets of 2-4 examples that are all primes (e.g. 67,31,17,61; 3,7; 59,3), yes rate on prime non-example targets minus non-prime odd targets; observed above null_mean means people pick up the "primes" rule more than the model predicts, below means the model over-generalizes along primes.
def test_statistic(df):
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97], dtype=float)
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    allp = np.all(np.isin(ex, primes) | np.isnan(ex), axis=1) & (n >= 2)
    sel = allp & ~np.any(ex == t[:, None], axis=1)
    tp = np.isin(t, primes)
    a, b = sel & tp, sel & ~tp & (t % 2 == 1)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
