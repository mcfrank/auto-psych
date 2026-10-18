# name: prime_target_boost_in_prime_sets
# description: In sets with >=2 examples, all prime, yes-rate on non-example prime targets minus odd non-prime targets; observed above model mean means people infer a 'primes' rule that the digit model (which only sees final digit/parity) misses.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    def isprime(a):
        a = np.nan_to_num(a).astype(int)
        out = a >= 2
        for p in (2, 3, 5, 7):
            out &= ~((a % p == 0) & (a != p))
        return out
    allp = ((isprime(X) | ~valid).all(axis=1)) & (n >= 2)
    inset = (X == t[:, None]).any(axis=1)
    tp = isprime(t); odd = (t.astype(int) % 2 == 1)
    a = allp & ~inset & tp; b = allp & ~inset & ~tp & odd
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
