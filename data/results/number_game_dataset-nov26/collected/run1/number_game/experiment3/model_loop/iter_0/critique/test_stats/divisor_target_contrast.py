# name: divisor_target_contrast
# description: Among targets >=10 from every example and sharing no example's final digit, yes-rate for targets that divide some example (incl. 1) minus yes-rate for targets that are not multiples, divisors or gcd>=3 kin of any example; observed > null means people generalize to divisors/factors of the examples (e.g. 15,11 -> 1,3,5) which the model has no route for, observed < null means the model over-credits them.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    dist = np.abs(t - e).min(1)
    dig = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    isex = (t == e).any(1)
    mult = (pres & (e1 > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    divi = (pres & (np.mod(e1, t) == 0) & (t != e)).any(1)
    kin = (pres & (np.gcd(t.astype(int), e1.astype(int)) >= 3)).any(1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    base = (dist >= 10) & ~dig & ~isex
    return rate(base & divi & ~mult) - rate(base & ~mult & ~divi & ~kin)
