# name: prime_set_prime_target_advantage
# description: In sets of >=2 examples that are all primes, yes rate for odd prime targets minus yes rate for odd composite targets; t_observed > null_mean means the model under-produces "prime number" rule generalization (it only sees parity/final digit).
def test_statistic(df):
    def isp(n):
        if n < 2:
            return False
        for k in range(2, int(n ** 0.5) + 1):
            if n % k == 0:
                return False
        return True
    primes = np.array([isp(i) for i in range(0, 101)])
    sets = df["number_set"].astype(str)
    flag = {}
    for s in sets.unique():
        xs = [int(x) for x in s.split(",")]
        flag[s] = len(xs) >= 2 and all(primes[x] for x in xs)
    ps = sets.map(flag).astype(bool).values
    t = df["target"].astype(int).values
    tp = primes[t]
    odd = (t % 2 == 1) & (t > 1)
    r = df["rating"].values.astype(float)
    a = ps & tp & odd
    b = ps & ~tp & odd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
