# name: digit_match_near_minus_far
# description: Among targets sharing a final digit with at least one example, yes rate when the nearest example is within 10 minus yes rate when it is more than 30 away; observed above null means numerical proximity boosts digit-matched targets, which the magnitude-blind model cannot capture.
def test_statistic(df):
    s = df["number_set"].astype(str)
    parts = s.str.split(",", expand=True)
    X = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    valid = ~np.isnan(X)
    Xf = np.where(valid, X, -1000.0)
    dist = np.abs(Xf - t[:, None])
    dist = np.where(valid, dist, np.inf)
    mind = dist.min(axis=1)
    match = (valid & ((np.nan_to_num(X, nan=-1).astype(int) % 10) == (t % 10)[:, None])).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    a = r[match & (mind <= 10)]
    b = r[match & (mind > 30)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
