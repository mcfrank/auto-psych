# name: odd_set_prime_minus_nonprime_yes
# description: Among sets of >= 2 examples that are all odd, yes rate for odd prime non-example targets minus odd non-prime non-example targets; observed > null means people pick up primality more than the model, observed < null means the model overweights the primes rule relative to people.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    nonex = dist > 0
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    odd_set = (n >= 2) & (np.nansum(np.where(np.isnan(ex), 0, ex % 2), axis=1) == n)
    ti = t.astype(int)
    primes = np.array([k > 1 and all(k % a for a in range(2, int(k ** 0.5) + 1)) for k in range(101)])
    pr = primes[ti]
    todd = ti % 2 == 1
    return rate(odd_set & nonex & todd & pr) - rate(odd_set & nonex & todd & ~pr)
