# name: interpolation_vs_extrapolation_contrast
# description: In sets of >=2 examples, yes-rate for targets lying strictly between the smallest and largest example minus targets outside that range, both 5-20 from the nearest example with no shared final digit and no multiple/divisor/gcd>=3 kinship; observed > null means people fill in the set's range (a region/interval concept) more than the model's sum of per-example similarities predicts, observed < null means they extrapolate more.
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
    base = (n >= 2) & (dist >= 5) & (dist <= 20) & ~dig & ~mult & ~divi & ~kin & ~isex
    inside = (t1 > emin) & (t1 < emax)
    return rate(base & inside) - rate(base & ~inside)
