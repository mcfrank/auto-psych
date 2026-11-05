# name: prime_target_in_prime_set_contrast
# description: In all-odd sets of >=2 examples where at least two examples are prime (>2), yes-rate for prime targets minus odd composite targets, both >=10 from the nearest example with no shared final digit; observed > null means people induce a 'prime numbers' concept the similarity model lacks, observed < null means primes are disfavored relative to the model.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d_all = np.abs(t - e)
    dist = d_all.min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    emin = np.where(pres, ex, 1e9).min(1)
    emax = np.where(pres, ex, -1e9).max(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    primes = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97], dtype=float)
    isp_e = np.isin(e1, primes) & pres & (e1 > 2)
    isp_t = np.isin(t1, primes)
    par = np.where(pres, np.mod(e1, 2), 1.0)
    allodd = (par == 1).all(1)
    base = allodd & (n >= 2) & (isp_e.sum(1) >= 2) & (dist >= 10) & ~dig & ~isex & (np.mod(t1, 2) == 1) & (t1 > 1)
    return rate(base & isp_t) - rate(base & ~isp_t)
