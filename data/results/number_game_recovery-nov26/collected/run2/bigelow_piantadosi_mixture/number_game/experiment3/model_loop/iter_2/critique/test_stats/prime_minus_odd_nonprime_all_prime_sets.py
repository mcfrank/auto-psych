# name: prime_minus_odd_nonprime_all_prime_sets
# description: Among sets of 2+ examples that are all prime, yes rate on prime non-example targets minus yes rate on odd non-prime targets; observed above null means people discriminate primes from merely-odd numbers more sharply than the model predicts.
def test_statistic(df):
    x = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    n = x.notna().sum(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    isin = x.eq(t, axis=0).any(axis=1)
    dist = x.sub(t, axis=0).abs()
    nd = dist.min(axis=1)
    P = [p for p in range(2, 101) if all(p % a for a in range(2, int(p ** 0.5) + 1))]
    allp = x.isin(P).sum(axis=1).eq(n) & (n >= 2)
    tp = t.isin(P)
    odd = (t % 2 == 1)
    v = r[allp & tp & ~isin].mean() - r[allp & odd & ~tp].mean()
    return float(0.0 if (v is None or np.isnan(v)) else v)
