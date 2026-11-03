# name: prime_set_prime_target_effect
# description: Among sets of >=2 examples that are all odd primes, yes rate for odd prime non-example targets minus yes rate for odd composite targets; observed > null means people pick up the "primes" concept more than the model predicts, observed < null means the model over-uses the primes rule relative to people (who treat such sets as odd numbers or ranges).
def test_statistic(df):
    pr = np.zeros(101, dtype=bool)
    for n in range(2, 101):
        pr[n] = all(n % a for a in range(2, int(n ** 0.5) + 1))
    sets = df["number_set"].astype(str)
    ex = sets.str.split(",", expand=True).reindex(columns=range(4)).fillna(0).astype(int).values
    has = ex > 0
    nex = has.sum(1)
    t = df["target"].astype(int).values
    allp = (np.where(has, pr[ex] & (ex % 2 == 1), True)).all(1)
    isex = ((ex == t[:, None]) & has).any(1)
    base = (nex >= 2) & allp & ~isex & (t % 2 == 1) & (t > 1)
    r = df["rating"].values.astype(float)
    a, b = base & pr[t], base & ~pr[t]
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
