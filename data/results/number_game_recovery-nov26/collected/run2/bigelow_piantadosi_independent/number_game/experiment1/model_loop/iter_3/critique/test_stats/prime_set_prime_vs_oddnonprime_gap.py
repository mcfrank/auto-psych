# name: prime_set_prime_vs_oddnonprime_gap
# description: In sets of >=2 examples that are all primes, yes rate for prime non-example targets minus for odd non-prime targets; t_observed above null_mean means people use the "primes" rule more than the model, below means less (they treat such sets as merely odd/near).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].to_numpy(dtype=float)
    sv = np.where(v, s, np.nan)
    is_ex = (np.where(v, s, -999) == t[:, None]).any(axis=1)
    d = np.nanmin(np.abs(sv - t[:, None]), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    pr = np.zeros(102, dtype=bool)
    for k in range(2, 101):
        pr[k] = all(k % a for a in range(2, int(k ** 0.5) + 1))
    si = np.where(v, s, 0).astype(int)
    allp = np.where(v, pr[si], True).all(axis=1)
    ti = t.astype(int)
    base = (n >= 2) & allp & ~is_ex
    return rate(base & pr[ti]) - rate(base & ~pr[ti] & (ti % 2 == 1))
