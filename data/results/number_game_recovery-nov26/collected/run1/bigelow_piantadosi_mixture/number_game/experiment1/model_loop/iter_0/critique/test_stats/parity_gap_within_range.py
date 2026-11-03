# name: parity_gap_within_range
# description: Among sets whose examples all share one parity, for targets strictly inside the example range (not equal to an example), yes-rate for parity-matching minus mismatching targets; observed above null means humans weight parity more than the model, which lets intervals say yes regardless of parity.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    dist = np.nanmin(np.abs(X - t[:, None]), axis=1)
    n = np.sum(~np.isnan(X), axis=1)
    inr = (t >= lo) & (t <= hi)
    par = np.nan_to_num(X % 2, nan=-1)
    valid = ~np.isnan(X)
    p0 = np.where(valid, par, np.nan)
    same = np.nanmin(p0, axis=1) == np.nanmax(p0, axis=1)
    match = (t % 2) == np.nanmin(p0, axis=1)
    base = same & inr & (dist > 0)
    a = base & match; b = base & ~match
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
