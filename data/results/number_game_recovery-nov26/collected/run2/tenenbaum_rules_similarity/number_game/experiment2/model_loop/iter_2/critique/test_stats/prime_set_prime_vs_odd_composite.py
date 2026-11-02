# name: prime_set_prime_vs_odd_composite
# description: Among sets of 2+ examples that are all primes, yes rate for non-example prime targets minus yes rate for odd composite targets; observed above the null means people use the prime concept more sharply than the model's prime-vs-odd weighting implies (below = people treat all-prime sets as merely odd more than the model does).
def test_statistic(df):
    primes = set(n for n in range(2, 101) if all(n % a for a in range(2, int(n ** 0.5) + 1)))
    isp = np.array([v in primes for v in range(101)])
    sets = df["number_set"].astype(str)
    ok = {}
    for s in sets.unique():
        ex = [int(x) for x in s.split(",")]
        ok[s] = len(ex) >= 2 and all(e in primes for e in ex)
    keep = sets.map(ok).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    exm = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    inset = (exm == df["target"].astype(int).to_numpy()[:, None]).any(axis=1)
    base = keep & ~inset
    a = base & isp[t]
    b = base & ~isp[t] & (t % 2 == 1) & (t > 1)
    r = df["rating"].to_numpy()
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
