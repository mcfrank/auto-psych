# name: inside_range_gap_vs_near
# description: Multi-example sets with span >=30: yes-rate for inside-range non-example targets >=10 from the nearest example minus those <=3 from it (targets sharing a last digit with any example excluded); observed below null_mean means people generalise by similarity to individual examples more than the model's flat interval fill, above means people fill gaps more evenly than the model.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    shares = np.any(ld == np.mod(t, 10)[:, None], axis=1)
    base = (n >= 2) & (hi - lo >= 30) & (t > lo) & (t < hi) & ~isex & ~shares
    a = base & (dmin >= 10); b = base & (dmin <= 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
