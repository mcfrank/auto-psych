# name: all_round_set_multiple_of_five_contrast
# description: In sets whose examples all end in 0 or 5, yes-rate for non-example targets that are multiples of 5 minus targets that are not, both >=10 from the nearest example; observed > null means people induce a 'multiples of 5' (round-number) category from such sets more strongly than the model's digit/divisor/x10 routes predict, observed < null means the model over-generalizes to multiples of 5.
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
    rs = ((np.mod(e1, 5) == 0) | ~pres).all(1)
    base = rs & (dist >= 10) & ~isex
    m5 = np.mod(t1, 5) == 0
    return rate(base & m5) - rate(base & ~m5)
