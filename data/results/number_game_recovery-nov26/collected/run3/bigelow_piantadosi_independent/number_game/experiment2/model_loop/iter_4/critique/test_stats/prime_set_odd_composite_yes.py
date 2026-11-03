# name: prime_set_odd_composite_yes
# description: In sets of >=2 examples that are all prime, yes rate on odd composite (non-prime, >1) targets; observed > null means the model's prime-over-odd preference suppresses "odd" readings too strongly after prime sets (observed < null: not strongly enough).
def test_statistic(df):
    def isp(n):
        return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {s: (lambda ex: len(ex) >= 2 and all(isp(v) for v in ex))([int(v) for v in s.split(",")]) for s in sets.unique()}
    ok = sets.map(feat).to_numpy(bool)
    primes = np.array([isp(n) for n in range(101)])
    m = ok & (t % 2 == 1) & (t > 1) & ~primes[t]
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
