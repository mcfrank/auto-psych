# name: set_size_mid_distance_contrast
# description: Among targets 3-12 from the nearest example that share no example's final digit and are not multiples or divisors of any example, yes-rate in one-example sets minus yes-rate in three- or four-example sets; summed similarity makes the model expect more generalization with more examples, so observed > null means people narrow with more examples (size-principle tightening) more than the model predicts, observed < null means they broaden more.
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

    base = (d >= 3) & (d <= 12) & nodig & ~mult & ~divi
    a, b = base & (n == 1), base & (n >= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
