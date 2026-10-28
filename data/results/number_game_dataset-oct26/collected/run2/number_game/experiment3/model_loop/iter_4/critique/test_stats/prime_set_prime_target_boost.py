# name: prime_set_prime_target_boost
# description: Among sets of 2+ examples that are all prime, yes-rate on prime non-example targets minus odd non-prime targets (parity-matched control); observed > null means people infer a "prime numbers" concept that the model (no prime hypothesis) under-produces.
def test_statistic(df):
    primes = [n for n in range(2, 101) if all(n % p for p in range(2, int(n ** 0.5) + 1))]
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    k = ex.notna().sum(axis=1).values
    allp = (ex.isin(primes) | ex.isna()).all(axis=1).values & (k >= 2)
    isex = (ex.values == t.values[:, None]).any(axis=1)
    tp = t.isin(primes).values
    odd = (t.values % 2 == 1)
    r = df["rating"].values.astype(float)
    a = allp & ~isex & tp; b = allp & ~isex & ~tp & odd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
