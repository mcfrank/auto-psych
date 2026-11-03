# name: same_decade_near_gap
# description: Among non-example targets 1-9 from the nearest example, yes rate when the target shares its tens digit (decade, n//10) with some example minus when it does not; t_observed above null_mean means people generalize within decades more than the model's distance-only intervals predict.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s)
    t = df["target"].astype(int).to_numpy().astype(float)
    dist = np.nanmin(np.abs(s - t[:, None]), axis=1)
    sd = (np.where(v, s // 10, -99) == (t // 10)[:, None]).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    base = (dist >= 1) & (dist <= 9)
    a = base & sd; b = base & ~sd
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
