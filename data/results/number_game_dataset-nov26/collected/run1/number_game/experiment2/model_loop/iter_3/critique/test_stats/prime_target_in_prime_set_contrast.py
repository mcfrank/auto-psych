# name: prime_target_in_prime_set_contrast
# description: In sets of >=2 examples that are all primes, yes-rate for prime targets minus yes-rate for odd non-prime targets (target not itself an example); observed > null means people induce a "prime numbers" rule that the similarity model (no primality dimension) under-produces, observed < null means primes are over-endorsed by the model.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    g = np.gcd(t.astype(int), e1.astype(int))
    kin = (pres & (g >= 3)).any(1)

    primes = np.array([p for p in range(2, 101) if all(p % q for q in range(2, p))], dtype=float)
    exz = np.where(pres, ex, 2.0)
    allp = np.isin(exz, primes).all(1) & (n >= 2)
    isex = (pres & (t == e)).any(1)
    tp = np.isin(t1, primes)
    a = allp & tp & ~isex
    b = allp & ~tp & (np.mod(t1, 2) == 1) & ~isex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
