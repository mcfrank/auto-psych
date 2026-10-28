# name: multi_outside_close_same_decade_gap
# description: For sets with 2+ examples, among targets outside the examples' span and 1-9 from the nearest example, yes-rate when the target shares a tens decade with that nearest example minus when it does not; observed above null means the decade grouping is still under-produced by the decade-bonus model, below means over-produced.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    dist = np.abs(ex - t[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    k = np.argmin(dist, axis=1)
    nearest = ex[np.arange(len(t)), k]
    dmin = dist[np.arange(len(t)), k]
    outside = (t < lo) | (t > hi)
    m = (n >= 2) & outside & (dmin >= 1) & (dmin <= 9)
    dec = lambda x: np.floor(np.minimum(x, 99) / 10)
    same = m & (dec(t) == dec(nearest))
    diff = m & (dec(t) != dec(nearest))
    if same.sum() == 0 or diff.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[diff].mean())
