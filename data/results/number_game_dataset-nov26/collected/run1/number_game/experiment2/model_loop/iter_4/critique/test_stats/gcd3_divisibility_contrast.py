# name: gcd3_divisibility_contrast
# description: In sets of 2+ examples whose greatest common divisor is exactly 3, yes-rate for multiple-of-3 targets minus non-multiples of 3, both restricted to targets sharing no example's final digit and not equal to an example; observed < null means people use 'multiple of 3' much more weakly than the model's single divisor penalty (shared with strict gcd-10 sets) implies, observed > null means more strongly.
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
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    ei = np.where(pres, ex, 0).astype(int)
    g = np.gcd.reduce(ei, axis=1)
    s = (g == 3) & (n >= 2)
    isex = ((t == e) & pres).any(1)
    base = s & ~digmatch & ~isex
    m3 = np.mod(t1, 3) == 0
    return diff(base & m3, base & ~m3)
