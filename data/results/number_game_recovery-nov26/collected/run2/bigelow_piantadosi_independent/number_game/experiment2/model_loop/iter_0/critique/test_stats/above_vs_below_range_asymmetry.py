# name: above_vs_below_range_asymmetry
# description: Yes rate for targets 1-20 above the largest example minus targets 1-20 below the smallest example; observed above null means people extend generalization upward more than the symmetric-interval model predicts, below means downward.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    above = (t > hi) & (t - hi <= 20)
    below = (t < lo) & (lo - t <= 20)
    return rate(above) - rate(below)
