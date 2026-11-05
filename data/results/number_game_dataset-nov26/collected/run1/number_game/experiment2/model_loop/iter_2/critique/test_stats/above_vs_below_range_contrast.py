# name: above_vs_below_range_contrast
# description: Among targets outside the examples' range, 5-30 from the nearest example, sharing no example's final digit and not a multiple of any example, yes-rate for targets above the largest example minus targets below the smallest; observed > null means people extend the set upward more than downward (directional generalization) beyond the model's symmetric magnitude similarity, observed < null means downward extension is favoured.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    d = np.abs(t - e).min(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    lo = np.min(np.where(pres, ex, np.inf), axis=1)
    hi = np.max(np.where(pres, ex, -np.inf), axis=1)
    base = (d >= 5) & (d <= 30) & nodig & ~mult
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & (t1 > hi), base & (t1 < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
