# name: below_minus_above_range_near
# description: Yes-rate for targets 1-10 below the set minimum minus yes-rate for targets 1-10 above the set maximum; observed > null means people extend downward more (or upward less) than the model's symmetric interval prior predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    b = (t < lo) & (lo - t <= 10); a = (t > hi) & (t - hi <= 10)
    rb = r[b].mean() if b.any() else 0.0; ra = r[a].mean() if a.any() else 0.0
    return float(rb - ra)
