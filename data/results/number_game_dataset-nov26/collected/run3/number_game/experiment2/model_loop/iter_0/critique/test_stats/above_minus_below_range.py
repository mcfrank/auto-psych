# name: above_minus_below_range
# description: For sets of 2+ examples, yes rate of targets above the examples' max minus yes rate of targets below the examples' min; observed above null means people extrapolate upward more (and downward less) than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy().astype(float)
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    r = df["rating"].to_numpy().astype(float)
    above = (n_ex >= 2) & (t > hi)
    below = (n_ex >= 2) & (t < lo)
    if above.sum() == 0 or below.sum() == 0:
        return 0.0
    return float(r[above].mean() - r[below].mean())
