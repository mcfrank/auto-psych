# name: small_vs_large_magnitude_near_contrast
# description: Among targets 3-15 from their nearest example, sharing no example's final digit and not multiples of any example, yes-rate when that nearest example is <=30 minus when it is >=60; observed < null means a given numeric distance counts as more dissimilar among small numbers (log/Weber-compressed magnitude) than the model's linear distance predicts, observed > null means the opposite.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)[:, None]
    pres = ~np.isnan(ex)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    D = np.abs(t - e)
    j = np.argmin(D, axis=1)
    d = D[np.arange(len(j)), j]
    near_ex = e[np.arange(len(j)), j]
    nodig = ~((np.mod(t, 10) == np.mod(e, 10)) & pres).any(1)
    mult = (pres & (e > 1) & (np.mod(t, e1) == 0) & (t != e)).any(1)
    base = (d >= 3) & (d <= 15) & nodig & ~mult
    r = df["rating"].to_numpy(dtype=float)
    a, b = base & (near_ex <= 30), base & (near_ex >= 60)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
