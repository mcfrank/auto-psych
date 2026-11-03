# name: round_number_target_bonus
# description: For sets containing no multiple of 10, yes rate on non-example multiple-of-10 targets minus yes rate on targets 1-3 away from a multiple of 10, both within 15 of some example; observed above null means people favour round-number targets beyond what the model's concepts predict (below null: they disfavour them).
def test_statistic(df):
    parts = df["number_set"].str.split(",", expand=True).astype(float)
    vals = parts.values
    has10 = np.any(np.mod(vals, 10) == 0, axis=1)
    t = df["target"].astype(float).values
    dmin = np.nanmin(np.abs(vals - t[:, None]), axis=1)
    k = np.mod(t, 10)
    dist10 = np.minimum(k, 10 - k)
    base = (~has10) & (dmin >= 1) & (dmin <= 15)
    a = base & (dist10 == 0)
    b = base & (dist10 >= 1) & (dist10 <= 3)
    r = df["rating"].values.astype(float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
