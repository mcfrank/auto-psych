# name: multiset_factor_kin_contrast
# description: In sets with >=2 examples, among targets >=10 from every example and sharing no example's final digit and not a multiple or divisor of any example, yes-rate for targets sharing a factor >=3 with at least one example (factor-kin) minus targets sharing none; observed < null means the model's per-example factor-kin route (fit largely on lone examples) over-generalizes in multi-example sets, observed > null means people credit factor-kin even more there.
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

    base = (n >= 2) & (d >= 10) & nodig & ~mult & ~divi
    a, b = base & kin, base & ~kin
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
