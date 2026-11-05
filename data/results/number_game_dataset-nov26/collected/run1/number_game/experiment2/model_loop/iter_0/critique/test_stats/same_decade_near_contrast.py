# name: same_decade_near_contrast
# description: Among targets 1-5 away from their nearest example and sharing no example's final digit, yes-rate when the target is in the same decade (same tens digit) as that nearest example minus when it is in a different decade; observed > null means people use a decade/tens-digit category the model lacks, observed < null means decade boundaries matter less than distance-matched similarity predicts.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    dig = np.mod(ex, 10)
    d = np.abs(ex - t[:, None])
    d = np.where(np.isnan(d), 1e9, d)
    idx = d.argmin(1)
    near = ex[np.arange(len(t)), idx]
    dmin = d[np.arange(len(t)), idx]
    anydig = np.nansum((dig == np.mod(t, 10)[:, None]).astype(float), 1) > 0
    dec_t = np.minimum(t, 99) // 10; dec_e = np.minimum(near, 99) // 10
    m = (dmin >= 1) & (dmin <= 5) & ~anydig
    r = df["rating"].to_numpy().astype(float)
    a = m & (dec_t == dec_e); b = m & (dec_t != dec_e)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
