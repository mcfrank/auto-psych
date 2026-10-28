# name: small_vs_large_magnitude_near_gap
# description: Among non-member targets 4-15 away from their nearest example and sharing no final digit with any example, yes rate when the nearest example is >=50 minus when it is <=25; observed > model means similarity in size is log-like (Weber) — a fixed absolute gap counts as closer among large numbers — which the model's linear magnitude metric misses.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    y = df["rating"].astype(float).values
    valid = ~np.isnan(ex)
    d = np.where(valid, np.abs(t[:, None] - ex), np.inf)
    j = d.argmin(1)
    dn = d[np.arange(len(t)), j]
    xn = ex[np.arange(len(t)), j]
    digit = (valid & (np.nan_to_num(ex) % 10 == (t % 10)[:, None])).any(1)
    base = (dn >= 4) & (dn <= 15) & ~digit
    a = base & (xn >= 50)
    b = base & (xn <= 25)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
