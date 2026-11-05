# name: large_example_multiple_contrast
# description: Among targets >=10 from every example and sharing no example's final digit, yes-rate for targets that are multiples only of an example >=13 (no example 2-12 divides them; e.g. 94 after 47) minus yes-rate for targets that are not multiples, divisors or factor-kin of any example; observed < null means the model's size-independent multiples route over-credits multiples of large examples (people's multiple route is a times-table effect for small numbers), observed > null means it under-credits them.
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

    small = pres & (e1 >= 2) & (e1 <= 12) & (np.mod(t, e1) == 0) & (t != e)
    large = pres & (e1 >= 13) & (np.mod(t, e1) == 0) & (t != e)
    base = (d >= 10) & nodig
    a = base & large.any(1) & ~small.any(1)
    b = base & ~mult & ~divi & ~kin
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
