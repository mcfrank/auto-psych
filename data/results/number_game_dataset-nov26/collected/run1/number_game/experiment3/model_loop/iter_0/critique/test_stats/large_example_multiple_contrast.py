# name: large_example_multiple_contrast
# description: Among targets >=10 from every example sharing no example's final digit, yes-rate for multiples of some example >=13 (but of no example 2-12) minus yes-rate for unrelated targets (not multiple/divisor/gcd>=3 kin); observed < null means the model's size-independent multiples route over-credits multiples of two-digit examples (people use only times-table multiples), observed > null means it under-credits them.
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
    small = (pres & (e1 >= 2) & (e1 <= 12) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    large = (pres & (e1 >= 13) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    base = (dist >= 10) & ~dig
    return rate(base & large & ~small) - rate(base & ~mult & ~divi & ~kin)
