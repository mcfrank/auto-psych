# name: yes_rate_far_targets
# description: Yes rate for targets more than 40 away from the nearest end of the examples' [min, max] range; observed above the null means humans have a higher floor of far-generalization (or lapse) "yes" responses than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    lo, hi = np.nanmin(sets, axis=1), np.nanmax(sets, axis=1)
    t = df["target"].astype(float).values
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    mask = dist > 40
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
