# name: just_outside_range_yes_rate
# description: Among multi-example sets, yes rate on targets lying 1-5 outside the examples' min-max range; observed above null means people extend ranges beyond the examples more than the model's interval hypotheses predict, below null means sharper edges.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.isfinite(ex).sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    m = (n >= 2) & (dist >= 1) & (dist <= 5)
    return float(y[m].mean()) if m.sum() > 0 else 0.0
