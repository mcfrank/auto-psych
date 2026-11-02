# name: decade_boundary_effect
# description: In 2-4 example sets, among non-example targets whose final digit matches no example and whose distance to the nearest example is 4-9, yes-rate for targets sharing a tens digit (decade) with some example minus those sharing none; observed above the null means people use decade membership beyond numeric distance more than the model (rules + intervals) predicts, below means the model over-produces a decade effect.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    nondig = ~is_ex & ~np.any(Dg == np.mod(t, 10)[:, None], axis=1)
    dec = np.any(np.floor(E / 10) == np.floor(t / 10)[:, None], axis=1)
    base = (n >= 2) & nondig & (dist >= 4) & (dist <= 9)
    a = base & dec
    b = base & ~dec
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
