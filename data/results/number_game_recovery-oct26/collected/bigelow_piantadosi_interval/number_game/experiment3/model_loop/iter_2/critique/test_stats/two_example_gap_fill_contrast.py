# name: two_example_gap_fill_contrast
# description: For 2-example sets whose examples are at least 15 apart, yes rate on non-example targets strictly between them minus targets outside their range but within the same distance (the gap) of the nearer example; observed above null_mean means people fill in the interval between two examples more (relative to extrapolating beyond) than the model predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    gap = hi - lo
    base = (n == 2) & (gap >= 15) & ~np.any(ex == t[:, None], axis=1)
    inside = base & (t > lo) & (t < hi)
    dout = np.where(t < lo, lo - t, t - hi)
    outside = base & ((t < lo) | (t > hi)) & (dout <= gap)
    if not inside.any() or not outside.any():
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
