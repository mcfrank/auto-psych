# name: yes_rate_near_outside_range
# description: Yes rate for targets lying outside the examples' [min, max] range but within 10 of it (multi-example sets only); observed above the null means humans generalize beyond the examples' range more than the size-principle intervals predict.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    n = (~np.isnan(sets)).sum(axis=1)
    lo, hi = np.nanmin(sets, axis=1), np.nanmax(sets, axis=1)
    t = df["target"].astype(float).values
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    mask = (dist > 0) & (dist <= 10) & (n >= 2)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
