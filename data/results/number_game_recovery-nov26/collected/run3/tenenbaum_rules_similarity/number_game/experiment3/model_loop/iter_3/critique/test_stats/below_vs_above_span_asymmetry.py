# name: below_vs_above_span_asymmetry
# description: For sets of >=2 examples, yes rate for targets 1-15 below the smallest example minus yes rate for targets 1-15 above the largest; the model's ranges are symmetric, so observed above (below) null means people extend concepts downward (upward) more than predicted.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = (~np.isnan(ex)).sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    below = (n >= 2) & (t < lo) & (lo - t <= 15)
    above = (n >= 2) & (t > hi) & (t - hi <= 15)
    if below.sum() == 0 or above.sum() == 0:
        return 0.0
    return float(y[below].mean() - y[above].mean())
