# name: same_digit_rule_distance_flatness
# description: In sets of >=2 examples that all share one final digit, yes-rate for non-example targets with that final digit lying >=20 from the nearest example minus those <=10 from it; observed > null (closer to 0) means people apply the final-digit rule flatly regardless of distance, while the model still decays digit-matching support with magnitude distance; observed < null means people's digit generalization is more local than the model's.
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
    ud = np.where(pres, np.mod(e1, 10), np.nan)
    same = (np.nanmax(ud, 1) == np.nanmin(ud, 1)) & (n >= 2)
    base = same & dig & ~isex
    return rate(base & (dist >= 20)) - rate(base & (dist <= 10))
