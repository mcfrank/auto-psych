# name: decade_match_near_contrast
# description: Among targets 2-9 away from the nearest example that share no example's final digit and are not multiples of any example, yes-rate for targets in the same decade (same tens digit) as some example minus those in no example's decade; observed > null means people use categorical decade membership beyond the model's linear magnitude distance (model under-produces same-decade generalization), observed < null means decade boundaries matter less than the model's distance implies.
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

    dec = (pres & (np.floor(t / 10) == np.floor(e / 10))).any(1)
    base = (d >= 2) & (d <= 9) & nodig & ~mult
    a, b = base & dec, base & ~dec
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
