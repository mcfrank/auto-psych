# name: last_digit_gap_mixed_sets
# description: Among sets with mixed parity (no parity or ends-in rule fits), for non-example targets, yes-rate when target shares its last digit with some example minus when it does not; observed above null means humans generalise by shared last digit more than the model predicts.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    valid = ~np.isnan(X)
    p0 = np.where(valid, X % 2, np.nan)
    mixed = np.nanmin(p0, axis=1) != np.nanmax(p0, axis=1)
    ld = np.any(valid & ((np.nan_to_num(X) % 10) == (t % 10)[:, None]), axis=1)
    base = mixed & (dist > 0)
    a = base & ld; b = base & ~ld
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
