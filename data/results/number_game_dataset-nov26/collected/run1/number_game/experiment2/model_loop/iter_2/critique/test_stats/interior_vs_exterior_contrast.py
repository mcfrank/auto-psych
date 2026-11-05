# name: interior_vs_exterior_contrast
# description: In sets with >=2 examples, among targets 4-20 from the nearest example that share no example's final digit and are not multiples of any example, yes-rate for targets strictly inside the examples' range minus targets outside it; observed > null means people fill in the interval spanned by the examples (range/hull generalization) more than summed exemplar similarity predicts, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    d = np.abs(t - e).min(1)
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    lo = np.nanmin(np.where(pres, ex, np.inf), axis=1)
    hi = np.nanmax(np.where(pres, ex, -np.inf), axis=1)
    inside = (t1 > lo) & (t1 < hi)
    base = (n >= 2) & (d >= 4) & (d <= 20) & nodig & ~mult
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & inside, base & ~inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
