# name: outside_range_distance_falloff
# description: Multi-example sets, non-example targets outside [min,max]: yes-rate for targets within 10 of the nearest example bound minus yes-rate for targets more than 25 beyond it; the model's hard bounds give a flat out-of-range penalty, so observed above null means people's generalization falls off gradually with distance outside the range (a graded magnitude band the model lacks).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    dist = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0.0))
    ok = (n >= 2) & (dist > 0)
    a = ok & (dist <= 10)
    b = ok & (dist > 25)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
