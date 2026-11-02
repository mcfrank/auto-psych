# name: prime_set_prime_target_advantage
# description: Among sets whose examples are all primes, yes-rate on odd prime targets minus odd non-prime targets (targets not in the set); observed above null means the model under-uses the "primes" rule (people discriminate primes from merely odd numbers more than predicted), below null means the model over-uses it.
def test_statistic(df):
    primes = np.array([n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1)) for n in range(101)])
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    exi = np.where(valid, ex, 2).astype(int)
    allprime = (primes[exi] | ~valid).all(axis=1)
    t = df["target"].astype(int).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    base = allprime & ~inset & (t % 2 == 1)
    r = df["rating"].to_numpy(dtype=float)
    a = base & primes[t]
    b = base & ~primes[t]
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
