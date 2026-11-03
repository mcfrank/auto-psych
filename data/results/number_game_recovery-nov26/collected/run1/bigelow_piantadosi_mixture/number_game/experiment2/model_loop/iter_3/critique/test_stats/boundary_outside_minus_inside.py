# name: boundary_outside_minus_inside
# description: Multi-example sets: yes-rate for non-example targets 1-5 outside the examples' [min,max] minus yes-rate for targets 1-5 inside the range from the nearest boundary; observed above null_mean means people's range boundary is softer (more extension past the extremes) than the model predicts, below means it is sharper.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dmin = np.nanmin(np.abs(E - t[:, None]), axis=1)
    ld = np.mod(E, 10)
    dist_b = np.minimum(np.abs(t - lo), np.abs(t - hi))
    inside = (t > lo) & (t < hi)
    outside = (t < lo) | (t > hi)
    base = (n >= 2) & ~isex & (dist_b >= 1) & (dist_b <= 5) & (hi - lo >= 12)
    a = base & outside; b = base & inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
