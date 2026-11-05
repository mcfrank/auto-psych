# name: all_prime_sets_prime_targets
# description: For sets of 2+ examples that are all prime (7,31; 2,47; 19,73; 67,31,17,61), yes rate on non-example prime targets; observed below null means the model over-trusts the "primes" rule (people barely use it), above = model under-generalizes to primes.
def test_statistic(df):
    primes = {p for p in range(2, 101) if all(p % a for a in range(2, int(p ** 0.5) + 1))}
    n = df["number_set"].str.count(",") + 1
    sub = df[n >= 2]
    parts = sub["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    isp = parts.isin(list(primes)) | parts.isna()
    allp = isp.all(axis=1)
    t = sub["target"].astype(int)
    is_ex = parts.eq(t, axis=0).any(axis=1)
    m = allp & t.isin(list(primes)) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(sub["rating"][m].mean())
