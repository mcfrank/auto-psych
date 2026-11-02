# name: above_vs_below_far
# description: Among targets 11-40 from the nearest example and outside the examples' range, yes rate above the max example minus below the min example; observed below null means the compressed-interval model over-produces far upward generalisation (humans are less asymmetric at long range), above null means humans extend far upward even more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    base = nonmember & (nearest > 10) & (nearest <= 40)
    a = r[base & (t > hi)]; b = r[base & (t < lo)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
