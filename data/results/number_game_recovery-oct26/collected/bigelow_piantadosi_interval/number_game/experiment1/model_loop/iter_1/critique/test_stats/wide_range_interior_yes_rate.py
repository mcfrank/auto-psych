# name: wide_range_interior_yes_rate
# description: Yes rate for non-example targets strictly inside the [min,max] of multi-example sets spanning at least 30; observed above the replicate mean means people interpolate (interval-like generalization) more than the model, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    m = (n >= 2) & (hi - lo >= 30) & (t > lo) & (t < hi) & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
