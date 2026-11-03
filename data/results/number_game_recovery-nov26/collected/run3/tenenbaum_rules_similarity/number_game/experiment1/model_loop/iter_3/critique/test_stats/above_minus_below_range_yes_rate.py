# name: above_minus_below_range_yes_rate
# description: For non-example targets outside the examples' [min,max] and within 15 of the nearest range edge, yes-rate for targets above the max minus targets below the min; the model's intervals extend symmetrically, so observed above model means people extrapolate upward (to larger numbers) more than downward, below means downward more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].to_numpy()
    above = (t > hi) & (t - hi <= 15)
    below = (t < lo) & (lo - t <= 15)
    if above.sum() == 0 or below.sum() == 0:
        return 0.0
    return float(r[above].mean() - r[below].mean())
