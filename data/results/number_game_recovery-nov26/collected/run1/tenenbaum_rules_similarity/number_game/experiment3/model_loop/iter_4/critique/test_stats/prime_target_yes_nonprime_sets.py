# name: prime_target_yes_nonprime_sets
# description: For sets with no prime example, yes rate for prime non-example targets minus non-prime non-example targets; observed above null means people over-endorse primes (a salient-category bias) the model does not produce, below null means they reject primes more than the model.
def test_statistic(df):
    pr = set(n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1)))
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    cache = {}
    for x in set(s):
        v = [int(a) for a in x.split(",")]
        cache[x] = (set(v), not any(a in pr for a in v))
    ex = np.array([ti in cache[x][0] for ti, x in zip(t, s)])
    np_set = np.array([cache[x][1] for x in s])
    isp = np.array([ti in pr for ti in t])
    a = y[np_set & ~ex & isp]
    b = y[np_set & ~ex & ~isp]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
