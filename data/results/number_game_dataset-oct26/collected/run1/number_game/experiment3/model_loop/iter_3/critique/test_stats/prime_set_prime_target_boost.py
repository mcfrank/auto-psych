# name: prime_set_prime_target_boost
# description: In multi-example sets whose examples are all primes, yes-rate on prime targets minus yes-rate on odd non-prime targets (targets equal to an example excluded); observed > null means the model under-produces a "prime numbers" concept, observed < null means it over-generalises to odd composites.
def test_statistic(df):
    P = {p for p in range(2, 101) if all(p % k for k in range(2, int(p ** 0.5) + 1))}
    sets = df["number_set"].astype(str)
    cats = pd.unique(sets)
    info = {}
    for s in cats:
        xs = [int(x) for x in s.split(",")]
        info[s] = (len(xs) >= 2 and all(x in P for x in xs), set(xs))
    ok = sets.map(lambda s: info[s][0]).values.astype(bool)
    t = df["target"].astype(int).values
    is_ex = np.array([ti in info[s][1] for s, ti in zip(sets.values, t)])
    isp = np.isin(t, list(P))
    r = df["rating"].values.astype(float)
    a = ok & ~is_ex & isp
    b = ok & ~is_ex & ~isp & (t % 2 == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
