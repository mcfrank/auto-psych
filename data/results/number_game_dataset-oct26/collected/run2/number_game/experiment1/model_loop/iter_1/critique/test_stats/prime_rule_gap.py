# name: prime_rule_gap
# description: Among sets of >=2 examples that are all prime, the yes rate for odd prime targets minus odd non-prime targets; observed > model means people pick up a "prime numbers" category that the exemplar similarity features (magnitude, digit, parity, divisibility) cannot represent.
def test_statistic(df):
    primes = set(p for p in range(2, 101) if all(p % q for q in range(2, int(p ** 0.5) + 1)))
    sets = df["number_set"].astype(str)
    allp = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        allp[s] = len(ex) >= 2 and all(x in primes for x in ex)
    m = sets.map(allp).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isp = np.isin(t, list(primes))
    odd = t % 2 == 1
    a, b = m & odd & isp, m & odd & ~isp
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
